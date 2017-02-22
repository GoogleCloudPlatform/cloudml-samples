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

"""This code implements a single node Feed forward neural network using TF
   low level APIs. It implements a binary classifier for Census Income Dataset.
"""


import argparse
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op
from tensorflow.contrib.layers.python.ops import bucketization_op

from StringIO import StringIO

import model
import os

#See tutorial on wide and deep https://www.tensorflow.org/tutorials/wide_and_deep/
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/feature_column.py

# csv columns in the input file
CSV_COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'gender',
               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
               'income_bracket')

CATEGORICAL_COLS = ('gender', 'race', 'education', 'marital_status',
                    'relationship', 'workclass','occupation', 'native_country')

CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week')

LABEL_COL = 'income_bracket'


#TODO: Change the gsutil to use the veneer storage
def read_local_or_gcs(file_name):
  """Read local or gcs file."""
  if file_name.startswith('gs://'):
    os.system('gsutil cp '+file_name+' '+os.path.basename(file_name))
    return open(os.path.basename(file_name)).read()
  else:
    local_file = open(file_name, 'r')
    return local_file.read()

def get_inputs_and_labels():
  """Placeholder for inputs and labels
     input shape = [None, 364] label shape = [None, 2]."""
  inputs = tf.placeholder(tf.float32, shape=[None, 346])
  labels = tf.placeholder(tf.float32, shape=[None, 2])
  return (inputs, labels)

def read_input_data(file_name, skiprows=None):
  """Read the input data as a pandas DataFrame of features and labels."""
  input_df = pd.read_csv(StringIO(read_local_or_gcs(file_name)), names=CSV_COLUMNS, skiprows=skiprows)

  label_df = input_df.pop(LABEL_COL)
  return (input_df, label_df)


def generate_input(input_df, label_df):
  """Prepare the input columns using SparseTensor."""

  # convert the continuous columns into tf.constant tensor
  continuous_columns = [
      tf.constant(input_df[col].values) for col in CONTINUOUS_COLS
  ]

  # convert the categorical columns into sparse tensors
  categorical_columns = [
      tf.SparseTensor(
          indices=[[i, 0] for i in range(input_df[col].size)],
          values=input_df[col].astype('category').cat.codes.values,
          dense_shape=[input_df[col].size, 1])
      for col in CATEGORICAL_COLS
  ]

  # convert the labels into one hot encoding
  label_tensor = tf.one_hot(
      label_df.astype('category').cat.codes.values,
      2, off_value=1, on_value=0)

  return (
      continuous_columns + categorical_columns, label_tensor
  )


def sparse_cross(feature_tensors, num_buckets, name='cross'):
  """Sparse feature cross of the feature SparseTensors."""
  return sparse_feature_cross_op.sparse_feature_cross(
      feature_tensors,
      hashed_output=True,
      num_buckets=num_buckets,
      hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY,
      name=name
  )


def generate_wide_columns(input_columns):
  """Generate wide columns by adding feature crosses of SparseTensors."""
  (age, education_num, capital_gain, capital_loss, hours_per_week,
   gender, race, education, marital_status, relationship, workclass,
   occupation, native_country) = input_columns

  # bucketize the age feature
  # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/ops/bucketization_op.py
  age_bucket = bucketization_op.bucketize(age, [18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  wide_columns = [
      sparse_cross([education, occupation], 15 * 16, name='edu_occ'),
      sparse_cross(
        [age_bucket, race, occupation],
        10 * 5 * 15,
        name='age_race_occ'),
      sparse_cross(
        [native_country, occupation],
        42 * 15,
        name='native_country_occ'),
      gender,
      native_country,
      education,
      occupation,
      workclass,
      marital_status,
      relationship,
      age_bucket
  ]

  return wide_columns


def concat_wide_columns(wide_columns):
  """Concat the tensors from wide columns."""

  (edu_occ, age_race_occ, native_country_occ, gender,
   native_country, education, occupation, workclass,
   marital_status, relationship, age_bucket) = wide_columns

  dense_tensors = [
      sparse_to_dense(edu_occ, 15 * 16),
      sparse_to_dense(gender, 2),
      sparse_to_dense(workclass, 9),
      sparse_to_dense(native_country, 42),
      sparse_to_dense(education, 16),
      sparse_to_dense(occupation, 15),
      sparse_to_dense(workclass, 9),
      sparse_to_dense(marital_status, 7),
      sparse_to_dense(relationship, 6)
  ]

  return tf.concat(dense_tensors, 1)


def sparse_to_dense(sparse_tensor, vocab_size):
  """Convert the sparse to dense tensor."""
  dense_tensor = tf.sparse_to_indicator(sparse_tensor, vocab_size)
  dense_tensor = tf.cast(dense_tensor, tf.int32)
  return dense_tensor

def read_input_tensor(input_file, skiprows=None):
  """Concatenate the wide columns to produce a single tensor."""
  inp, label = read_input_data(input_file, skiprows)
  in_tensor, label_tensor = generate_input(inp, label)
  return concat_wide_columns(generate_wide_columns(in_tensor)), label_tensor

def loss(model, labels):
  """Compute cross entropy loss function."""
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=labels))
  return cross_entropy

def optimizer(loss, global_step):
  """Gradient descent optimizer with 0.5 learning rate."""
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
      loss, global_step=global_step)
  return train_step


def training_steps(session, train_step, model, labels, global_step, max_steps,
                   inp, label,
                   eval_inp, eval_label):
  """Run the training steps and calculate accuracy every 10 steps."""
  step = tf.train.global_step(session, global_step)

  while step < max_steps:
    session.run(
        train_step,
        feed_dict={
            inputs: session.run(inp),
            labels: session.run(label)
            }
    )

    step = tf.train.global_step(session, global_step)

    if step % 10 == 0:
      accuracy = evaluation(session, model, labels, eval_inp, eval_label)
      print('Step number {} of {} done, Accuracy {:.2f}%'.format(
          step, max_steps, accuracy))

  return train_step


def training_single(session, model, labels, max_steps,
                    inp_tensor, label_tensor,
                    eval_inp_tensor, eval_label_tensor):
  """Perform single node training."""

  global_step = tf.contrib.framework.get_or_create_global_step()
  init = tf.global_variables_initializer()
  session.run(init)

  cross_entropy = loss(model, labels)
  train_step = optimizer(cross_entropy, global_step)

  training_steps(session, train_step, model, labels, global_step, max_steps,
                 inp_tensor, label_tensor,
                 eval_inp_tensor, eval_label_tensor)

  return train_step

def training_distributed(session, model, labels, max_steps,
                         inp_tensor, label_tensor,
                         eval_inp_tensor, eval_label_tensor):
  """Perform distributed training."""

  cluster_spec, job_name, task_index = parse_tf_config()

  # /job:localhost/replica:0/task:0/cpu:0
  device_fn = tf.train.replica_device_setter(
      cluster=cluster_spec,
      ps_device="/job:localhost/replica:%d/task:%d/cpu:%d" % (task_index, task_index, task_index),
      #worker_device="/job:%s/task:%d" % (job_name, task_index)
      worker_device="/job:localhost/replica:%d/task:%d/cpu:%d" % (task_index, task_index, task_index)
  )

  # Create and start a server
  server = tf.train.Server(cluster_spec,
                           job_name=job_name,
                           task_index=task_index)

  if job_name == 'ps':
    server.join()
  elif job_name in ['master', 'worker']:
    with tf.device(device_fn):
      global_step = tf.contrib.framework.get_or_create_global_step()

      cross_entropy = loss(model, labels)
      train_step = optimizer(cross_entropy, global_step)

    init = tf.global_variables_initializer()
    session.run(init)

    training_steps(session, train_step, model, labels, global_step, max_steps,
                   inp_tensor, label_tensor,
                   eval_inp_tensor, eval_label_tensor)



def evaluation(session, model, labels, inp_tensor, label_tensor):
  """Perform the evaluation step to calculate accuracy."""
  correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return 100 * session.run(
      accuracy,
      feed_dict={
          inputs: session.run(inp_tensor),
          labels: session.run(label_tensor)
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

  print('cluster spec {} {} {}'.format(cluster, job_name, task_index))
  cluster_spec = tf.train.ClusterSpec(cluster)
  return cluster_spec, job_name, task_index


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_data_path', required=True, type=str,
      help='Training file location')
  parser.add_argument(
      '--eval_data_path', required=True, type=str,
      help='Evaluation file location')
  parser.add_argument(
      '--distributed', default=False, type=bool,
      help='Run the code either as single or distributed mode')
  parser.add_argument('--max_steps', type=int, default=200,
      help='Maximum number of training steps to perform')
  parse_args, unknown = parser.parse_known_args()

  train_tensor, train_lab_tensor = read_input_tensor(parse_args.train_data_path)
  eval_tensor, eval_lab_tensor = read_input_tensor(parse_args.eval_data_path,
                                                   skiprows=[0])

  session = tf.Session()

  inputs, labels = get_inputs_and_labels()
  nn_model = model.inference(inputs)

  # Start single node training
  if not parse_args.distributed:
    training_single(
        session, nn_model, labels,
        parse_args.max_steps, train_tensor, train_lab_tensor,
        eval_tensor, eval_lab_tensor
    )
  # Start distributed training
  elif parse_args.distributed:
    training_distributed(
        session, nn_model, labels,
        parse_args.max_steps, train_tensor, train_lab_tensor,
        eval_tensor, eval_lab_tensor
    )
