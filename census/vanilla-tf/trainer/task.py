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
from tensorflow.python.training import basic_session_run_hooks

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

CSV_COLUMN_DEFAULTS = [[0.], [''], [0.], [''], [0.], [''], [''], [''], [''], [''],
                                    [0.], [0.], [0.], [''], ['']]

# Categorical columns with vocab size
HASH_BUCKET_COLS = (('education', 16), ('marital_status', 7),
                    ('relationship', 6), ('workclass', 9),('occupation', 15), ('native_country', 42))
KEY_COLS = (('gender', ('female', 'male')), ('race', ('Amer-Indian-Eskimo',
                                                      'Asian-Pac-Islander',
                                                      'Black',
                                                      'Other',
                                                      'White')))


CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week')
CATEGORICAL_COLS = HASH_BUCKET_COLS + tuple((col, len(keys)) for col, keys in KEY_COLS)
LABELS = [' <=50K', ' >50K']
LABEL_COLUMN = 'income_bracket'

UNUSED_COLUMNS = set(CSV_COLUMNS) - set(
    zip(*CATEGORICAL_COLS)[0] + CONTINUOUS_COLS + (LABEL_COLUMN,))

EVAL = 'EVAL'
TRAIN = 'TRAIN'
## ***** TODO(puneith) I think everything between these markers should move to model.py

# Graph creation section for training and evaluation
def model_fn(inputs,
             labels,
             learning_rate=0.5,
             batch_size=40):
  """Create training and evaluation graph."""
  features = convert_sparse_columns(inputs)
  logits = model.dnn(tf.concat(features.values(), 1))

  global_step = tf.contrib.framework.get_or_create_global_step()
  cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.squeeze(labels)))

  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        cross_entropy, global_step=global_step)

  probabilities = tf.nn.softmax(logits)
  predictions = tf.argmax(probabilities, 1)
  accuracy_op, update_acc = tf.contrib.metrics.streaming_accuracy(predictions, labels)

  return train_op, accuracy_op, update_acc, global_step, predictions


def convert_sparse_columns(features):
  for col, bucket_size in HASH_BUCKET_COLS:
    features[col] = string_ops.string_to_hash_bucket_fast(
            features[col], bucket_size)
  
  for col, keys in KEY_COLS:
    table = tf.contrib.lookup.string_to_index_table_from_tensor(
        tf.constant(keys))
    features[col] = table.lookup(features[col])
  
  for col, size in CATEGORICAL_COLS:
    features[col] = tf.squeeze(tf.one_hot(
        features[col],
        size,
        axis=1,
        dtype=tf.float32), axis=[2])

  return features

# ***** TODO(puneith) here's the other marker
    
#
# Function to perform the actual training eval loop.
# This function is same for single and distributed.
#
def run(target,
        is_chief,
        max_steps,
        train_data_paths,
        eval_data_paths,
        job_dir,
        eval_every=100,
        eval_steps=10,
        learning_rate=0.1,
        num_epochs=None,
        batch_size=40):

  """Run the training steps and calculate accuracy every 10 steps."""

  training_eval_graph = tf.Graph()
  with training_eval_graph.as_default():
    with tf.device(tf.train.replica_device_setter()):
      mode = tf.placeholder(shape=[], dtype=tf.string)
      eval_features, eval_label = input_fn(
          eval_data_paths, shuffle=False, batch_size=batch_size)
      train_features, train_label = input_fn(
          train_data_paths, shuffle=True, num_epochs=num_epochs, batch_size=batch_size)

      is_train = tf.equal(mode, tf.constant(TRAIN))
      sorted_keys = train_features.keys()
      sorted_keys.sort()
      inputs = dict(zip(
          sorted_keys,
          tf.cond(
              is_train,
              lambda: [train_features[k] for k in sorted_keys],
              lambda: [eval_features[k] for k in sorted_keys]
          )
      ))
      labels = tf.cond(is_train, lambda: train_label, lambda: eval_label)
      train_op, accuracy_op, eval_op, global_step_tensor, predictions = model_fn(
          inputs, labels, learning_rate=learning_rate)

    with tf.train.MonitoredTrainingSession(master=target,
                                           is_chief=is_chief,
                                           checkpoint_dir=job_dir,
                                           save_checkpoint_secs=20,
                                           save_summaries_steps=50) as session:
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(coord=coord, sess=session)
      step = 0
      last_eval = 0 
      with coord.stop_on_exception():
        while step < max_steps and not coord.should_stop():
            if is_chief and step - last_eval > eval_every:
                last_eval = step
                for _ in range(eval_steps):
                    session.run([eval_op], feed_dict={mode: EVAL})
                accuracy = session.run([accuracy_op], feed_dict={mode: EVAL})
                print("Accuracy at step: {} is {}".format(step, accuracy))
            step, _ = session.run(
                [global_step_tensor, train_op],
                feed_dict={mode: TRAIN}
            )



def dispatch(*args, **kwargs):
  """Parse TF_CONFIG to cluster_spec, job_name and task_index."""

  tf_config = os.environ.get('TF_CONFIG')

  if not tf_config:
    return run('', True, *args, **kwargs)

  tf_config_json = json.loads(tf_config)

  cluster = tf_config_json.get('cluster')
  job_name = tf_config_json.get('task').get('type')
  task_index = tf_config_json.get('task').get('index')

  # If cluster information is empty run local
  if job_name is None or task_index is None:
    return run('', True, *args, **kwargs)

  cluster_spec = tf.train.ClusterSpec(cluster)
  server = tf.train.Server(cluster_spec,
                           job_name=job_name,
                           task_index=task_index)

  if job_name == 'ps':
    server.join()
    return
  elif job_name in ['master', 'worker']:
    return run(server.target, job_name == 'master')


def parse_label_column(label_string_tensor): 
  """Parses a string tensor into the label tensor
  Args:
    label_string_tensor: Tensor of dtype string. Result of parsing the
    CSV column specified by LABEL_COLUMN
  Returns:
    A Tensor of the same shape as label_string_tensor, should return
    an int64 Tensor representing the label index for classification tasks, 
    and a float32 Tensor representing the value for a regression task.
  """
  # Build a Hash Table inside the graph
  table = tf.contrib.lookup.string_to_index_table_from_tensor(
      tf.constant(LABELS))

  # Use the hash table to convert string labels to ints
  return table.lookup(label_string_tensor)


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=40):
  """Generates an input function for training or evaluation.
  Args:
      filenames: [str] list of CSV files to read data from.
      num_epochs: int how many times through to read the data.
        If None will loop through data indefinitely
      shuffle: bool, whether or not to randomize the order of data.
        Controls randomization of both file order and line order within
        files.
      skip_header_lines: int set to non-zero in order to skip header lines
        in CSV files.
      batch_size: int First dimension size of the Tensors returned by
        input_fn
  Returns:
      A function () -> (features, indices) where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
  """
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=shuffle)
  reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

  _, rows = reader.read_up_to(filename_queue, num_records=batch_size)
  
  # DNNLinearCombinedClassifier expects rank 2 tensors.
  row_columns = tf.expand_dims(rows, -1)
  columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))
  
  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)

  if shuffle:
    features = tf.train.shuffle_batch(
        features,
        batch_size,
        capacity=batch_size * 10,
        min_after_dequeue=batch_size*2 + 1,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
    )
  label_tensor = parse_label_column(features.pop(LABEL_COLUMN))
  return features, label_tensor


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_data_paths', required=True, type=str,
      help='Training file location', nargs='+')
  parser.add_argument(
      '--eval_data_paths', required=True, type=str,
      help='Evaluation file location', nargs='+')
  parser.add_argument(
      '--job_dir', required=True, type=str,
      help='Location to write checkpoints and export model'
  )
  parser.add_argument('--max_steps', type=int, default=1000,
      help='Maximum number of training steps to perform')
  parse_args, unknown = parser.parse_known_args()

  dispatch(**parse_args.__dict__)
