# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This code implements a Feed forward neural network using TF low level APIs.
   It implements a binary classifier for Census Income Dataset using both single
   and distributed node cluster.
"""


import argparse
import json
import tensorflow as tf
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op
from tensorflow.contrib.layers.python.ops import bucketization_op
from tensorflow.python.ops import string_ops

from StringIO import StringIO

import model
import os
import multiprocessing


tf.logging.set_verbosity(tf.logging.INFO)

#See tutorial on wide and deep https://www.tensorflow.org/tutorials/wide_and_deep/
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/feature_column.py

# csv columns in the input file
CSV_COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'gender',
               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
               'income_bracket')

DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                                    [0], [0], [0], [''], ['']]

# Categorical columns with vocab size
CATEGORICAL_COLS = (('gender', 2), ('race', 5), ('education', 16), ('marital_status', 7),
                    ('relationship', 6), ('workclass', 9),('occupation', 15), ('native_country', 42))

CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week')

LABEL_COLUMN = 'income_bracket'


#
# Graph creation section for training and evaluation
#
def make_graph(inputs, labels, learning_rate=0.5):
  """Create training and evaluation graph."""

  logits = model.inference(inputs)

  global_step = tf.contrib.framework.get_or_create_global_step()
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
      cross_entropy, global_step=global_step)

  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  eval_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return train_step, eval_step, global_step

def get_placeholders():
  """Create placeholder for inputs and prediction labels.

  Returns:
     tuple (tf.placeholder): 337 dimension feature and 2 classes
  """
  inputs = tf.placeholder(tf.float32, shape=[None, 337])
  labels = tf.placeholder(tf.float32, shape=[None, 2])
  return inputs, labels

def sparse_to_dense(sparse_tensor, vocab_size):
  """Convert the sparse to dense tensor."""
  dense_tensor = tf.sparse_to_indicator(sparse_tensor, vocab_size)
  dense_tensor = tf.cast(dense_tensor, tf.int32)
  return dense_tensor

#
# Feature crosses and generation of wide columns
#
def sparse_cross(feature_tensors, num_buckets, name='cross'):
  """Sparse feature cross of the feature SparseTensors."""
  return sparse_feature_cross_op.sparse_feature_cross(
      feature_tensors,
      hashed_output=True,
      num_buckets=num_buckets,
      hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY,
      name=name
  )

def to_continuous_and_categorical(features, labels, batch_size):
  """Convert the input columns to continuous and sparse tensor and
     labels to the one hot tensor.
  """

  # get the continuous columns
  continuous_columns = [
      features.get(col) for col in CONTINUOUS_COLS
  ]

  # convert the categorical columns into sparse tensors
  categorical_columns = [
      tf.SparseTensor(
          indices=[[i, 0] for i in range(batch_size)],
          values=string_ops.string_to_hash_bucket_fast(features[col], vocab),
          dense_shape=[batch_size, 1])
      for col, vocab in CATEGORICAL_COLS
  ]

  # convert the labels into one hot encoding
  one_hot_labels = tf.one_hot(labels, 2)

  return continuous_columns + categorical_columns, one_hot_labels


def generate_wide_columns(input_columns):
  """Generate wide columns by adding feature crosses of SparseTensors."""
  (age, education_num, capital_gain, capital_loss, hours_per_week,
   gender, race, education, marital_status, relationship, workclass,
   occupation, native_country) = input_columns

  # bucketize the age feature
  # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/ops/bucketization_op.py
  age_bucket = bucketization_op.bucketize(age, [18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  wide_columns = [
      (sparse_cross([education, occupation], 15 * 16, name='edu_occ'), 15 * 16),
      (gender, 2),
      (native_country, 42),
      (education, 16),
      (occupation, 15),
      (workclass, 9),
      (marital_status, 7),
      (relationship, 6)
  ]

  dense_tensors = [sparse_to_dense(col[0], col[1]) for col in wide_columns]

  return tf.concat(dense_tensors, 1)


#
# Function to perform the actual training loop.
# This function is same for single and distributed.
#
def training_steps(session, graph, inputs, labels,
                   max_steps, train_eval_tensor,
                   job_name='local', job_id=0):
  """Run the training steps and calculate accuracy every 10 steps."""

  train_step, eval_step, global_step = graph
  step = tf.train.global_step(session, global_step)

  (train_inp, train_label), (eval_inp, eval_label) = train_eval_tensor

  with session:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)

    while step < max_steps:
      session.run(
          train_step,
          feed_dict={
              inputs: session.run(train_inp),
              labels: session.run(train_label)
          }
      )
      step = tf.train.global_step(session, global_step)
      if step % 10 == 0:
        accuracy = evaluate_accuracy(session, eval_step, inputs, labels,
                                   eval_inp, eval_label)
        print('[{}/{}]: Step number {} of {} done, Accuracy {:.2f}%'.format(
            job_name, job_id, step, max_steps, accuracy))

    coord.request_stop()
    coord.join(threads)

  return train_step


#
# Single and Distributed training functions.
#
def training_single(job_dir, max_steps, train_eval_tensor):
  """Perform single node training."""

  inputs, labels = get_placeholders()

  graph = make_graph(inputs, labels)
  init = tf.global_variables_initializer()

  session = tf.train.MonitoredTrainingSession(checkpoint_dir=job_dir,
                                              save_checkpoint_secs=20,
                                              save_summaries_steps=50)
  session.run(init)

  training_steps(session, graph, inputs, labels, max_steps, train_eval_tensor)

def training_distributed(job_dir, max_steps, train_eval_tensor):
  """Perform distributed training."""

  # Parse the TF_CONFIG to create cluster spec and job name
  # We are doing this manually here to demonstrate
  # See RunConfig abstraction here https://goo.gl/f4BQMo
  cluster_spec, job_name, task_index = parse_tf_config()

  # Create and start a server
  server = tf.train.Server(cluster_spec,
                           job_name=job_name,
                           task_index=task_index)

  is_chief = (job_name == 'master')

  if job_name == 'ps':
    server.join()
  elif job_name in ['master', 'worker']:
    with tf.device(tf.train.replica_device_setter()):
      inputs, labels = get_placeholders()
      graph = make_graph(inputs, labels)

    init = tf.global_variables_initializer()

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           checkpoint_dir=job_dir,
                                           save_checkpoint_secs=20,
                                           save_summaries_steps=50) as session:
      session.run(init)
      training_steps(session, graph, inputs, labels, max_steps,
                     train_eval_tensor, job_name=job_name, job_id=task_index)


def evaluate_accuracy(session, eval_step, inputs, labels,
                      eval_input, eval_label):
  """Perform the evaluation step to calculate accuracy."""
  return 100 * session.run(
      eval_step,
      feed_dict={
          inputs: session.run(eval_input),
          labels: session.run(eval_label)
      }
  )


def parse_tf_config():
  """Parse TF_CONFIG to cluster_spec, job_name and task_index."""

  tf_config = os.environ.get('TF_CONFIG')

  if tf_config is None or tf_config == '':
    return None

  tf_config_json = json.loads(tf_config)

  cluster = tf_config_json.get('cluster')
  job_name = tf_config_json.get('task').get('type')
  task_index = tf_config_json.get('task').get('index')

  cluster_spec = tf.train.ClusterSpec(cluster)
  return cluster_spec, job_name, task_index


def read_input_tensors(file_name,
                       batch_size=40,
                       skiprows=None):
  """Read the input file as an input queue.
     See https://www.tensorflow.org/programmers_guide/reading_data/
  """

  filename_queue = tf.train.string_input_producer([file_name])

  reader = tf.TextLineReader(skip_header_lines=skiprows)
  key, value = reader.read(filename_queue)
  value_columns = tf.expand_dims(value, -1)

  columns = tf.decode_csv(
      value_columns,record_defaults=DEFAULTS)

  features = dict(zip(CSV_COLUMNS, columns))

  features = tf.train.shuffle_batch(
      features,
      batch_size,
      capacity=batch_size * 10,
      min_after_dequeue=batch_size*2 + 1,
      num_threads=multiprocessing.cpu_count(),
      enqueue_many=True,
      allow_smaller_final_batch=True
  )

  # Using tf.string_split because eval data has an extra "." in label column
  income_label = tf.to_int32(tf.equal(tf.string_split(
      features.pop(LABEL_COLUMN), '.').values, ' >50K'))

  features_list, income_class = to_continuous_and_categorical(
      features, income_label, batch_size)
  return generate_wide_columns(features_list), income_class


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_data_path', required=True, type=str,
      help='Training file location')
  parser.add_argument(
      '--eval_data_path', required=True, type=str,
      help='Evaluation file location')
  parser.add_argument(
      '--job_dir', required=True, type=str,
      help='Location to write checkpoints and export model'
  )
  parser.add_argument(
      '--distributed', default=False, type=bool,
      help='Run the code either as single or distributed mode')
  parser.add_argument('--max_steps', type=int, default=200,
      help='Maximum number of training steps to perform')
  parse_args, unknown = parser.parse_known_args()

  train_tensor = read_input_tensors(parse_args.train_data_path,
                                    batch_size=300)

  # Skip first row which has meta information
  eval_tensor = read_input_tensors(parse_args.eval_data_path,
                                   skiprows=1)

  train_eval_tensor = [train_tensor, eval_tensor]

  # Start single node training
  if not parse_args.distributed:
    training_single(
        parse_args.job_dir,
        parse_args.max_steps, train_eval_tensor
    )
  # Start distributed training
  elif parse_args.distributed:
    training_distributed(
        parse_args.job_dir,
        parse_args.max_steps, train_eval_tensor
    )
