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

STEPS = 100

def read_input_data(file_name):
  """Read the input data as a pandas DataFrame of features and labels."""
  input_df = pd.read_csv(file_name, names=CSV_COLUMNS)

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

# input training data with labels
train_input, train_label = read_input_data('widendeep/adult.data')
train_in_tensor, train_label_tensor = generate_input(train_input, train_label)

# input test data with labels
test_input, test_label = read_input_data('widendeep/adult.test')
test_in_tensor, test_label_tensor = generate_input(test_input, test_label)

train_tensor = concat_wide_columns(
    generate_wide_columns(train_in_tensor)
)

test_tensor = concat_wide_columns(
    generate_wide_columns(test_in_tensor)
)

sess = tf.Session()

inputs = tf.placeholder(tf.float32, shape=[None, 53])
labels = tf.placeholder(tf.float32, shape=[None, 2])

nn_model = model.inference(inputs)

init = tf.global_variables_initializer()
sess.run(init)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_model, labels=labels))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


for step in xrange(STEPS):
  sess.run(
      train_step,
      feed_dict={
          inputs: sess.run(train_tensor),
          labels: sess.run(train_label_tensor)
      }
  )

  if step % 10 == 0:
    print('Step number {} of {} done'.format(step, STEPS))

correct_prediction = tf.equal(tf.argmax(nn_model, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('\nAccuracy {0:.2f}%'.format(
    100 * sess.run(
        accuracy,
        feed_dict={
            inputs: sess.run(test_tensor),
            labels: sess.run(test_label_tensor)
        })))
