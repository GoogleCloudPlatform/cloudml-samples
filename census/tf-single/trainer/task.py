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


import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op
from tensorflow.contrib.layers.python.ops import bucketization_op

import model

#https://www.tensorflow.org/tutorials/wide_and_deep/
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

def read_input_data(file_name, skiprows=None):
  """Read the input data as a pandas DataFrame of features and labels."""
  input_df = pd.read_csv(file_name, names=CSV_COLUMNS, skiprows=skiprows)

  # replace missing values with np.nan and ignore
  #input_df = input_df.replace([' ?'], [np.nan])

  label_df = input_df.pop(LABEL_COL)
  return (input_df, label_df)


def generate_input(input_df, label_df):
  """Prepare the input columns using SparseTensor."""
  continuous_columns = [
      tf.constant(input_df[col].values) for col in CONTINUOUS_COLS
  ]

  categorical_columns = [
      tf.SparseTensor(
          indices=[[i, 0] for i in range(input_df[col].size)],
          values=input_df[col].astype('category').cat.codes.values,
          dense_shape=[input_df[col].size, 1])
      for col in CATEGORICAL_COLS
  ]

  sparse_t = tf.SparseTensor(
      indices=[[i, 0] for i in range(label_df.size)],
      values=label_df.astype('category').cat.codes.values,
      dense_shape=[label_df.size, 1]
  )

  dense_i_tensor = tf.sparse_to_indicator(sparse_t, 2)
  dense_i_tensor = tf.cast(dense_i_tensor, tf.int32)

  return (
      continuous_columns + categorical_columns, dense_i_tensor
  )


def sparse_cross(feature_tensors, name='cross'):
  """Sparse feature cross of the feature SparseTensors."""
  return sparse_feature_cross_op.sparse_feature_cross(
      feature_tensors,
      hashed_output=True,
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
      sparse_cross([education, occupation], name='edu_occ'),
      sparse_cross([age_bucket, race, occupation], name='age_race_occ'),
      sparse_cross([native_country, occupation], name='native_country_occ'),
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
      sparse_to_dense(gender, 2),
      sparse_to_dense(workclass, 9),
      sparse_to_dense(native_country, 42)
  ]

  return tf.concat(dense_tensors, 1)


def sparse_to_dense(sparse_tensor, vocab_size):
  """Convert the sparse to dense tensor."""
  dense_tensor = tf.sparse_to_indicator(sparse_tensor, vocab_size)
  dense_tensor = tf.cast(dense_tensor, tf.int32)
  return dense_tensor

def read_input_tensor(input_file, skiprows=None):
  print('skip rows ',skiprows)
  inp, label = read_input_data(input_file, skiprows)
  in_tensor, label_tensor = generate_input(inp, label)
  return concat_wide_columns(generate_wide_columns(in_tensor)), label_tensor

def training(session, model, labels, max_steps, inp_tensor, label_tensor):
  init = tf.global_variables_initializer()
  session.run(init)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=labels))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  for step in xrange(max_steps):
    session.run(
        train_step,
        feed_dict={
            inputs: session.run(inp_tensor),
            labels: session.run(label_tensor)
        }
    )
    if step % 10 == 0:
      print('Step number {} of {} done'.format(step, max_steps))

  return train_step

def evaluation(session, model, labels, inp_tensor, label_tensor):
  correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('\nAccuracy {0:.2f}%'.format(
      100 * session.run(
          accuracy,
          feed_dict={
              inputs: session.run(inp_tensor),
              labels: session.run(label_tensor)
          })))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_data_path', required=True, type=str)
  parser.add_argument(
      '--eval_data_path', required=True, type=str)
  parser.add_argument('--max_steps', type=int, default=300)
  parse_args = parser.parse_args()

  train_tensor, train_lab_tensor = read_input_tensor(parse_args.train_data_path)
  eval_tensor, eval_lab_tensor = read_input_tensor(parse_args.eval_data_path,
                                                   skiprows=[0])

  session = tf.Session()

  inputs = tf.placeholder(tf.float32, shape=[None, 53])
  labels = tf.placeholder(tf.float32, shape=[None, 2])
  nn_model = model.inference(inputs)

  training(
      session, nn_model, labels,
      parse_args.max_steps, train_tensor, train_lab_tensor
  )

  evaluation(
      session, nn_model, labels, eval_tensor, eval_lab_tensor
  )
