# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'gender',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
           'income_bracket']

LABEL_COLUMN = 'income_bracket'

DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
            [0], [0], [0], [''], ['']]


INPUT_COLUMNS = [
    layers.sparse_column_with_keys(
        column_name='gender', keys=['female', 'male']),

    layers.sparse_column_with_keys(
        column_name='race',
        keys=[
            'Amer-Indian-Eskimo',
            'Asian-Pac-Islander',
            'Black',
            'Other',
            'White'
        ]
    ),

    layers.sparse_column_with_hash_bucket(
        'education', hash_bucket_size=1000),
    layers.sparse_column_with_hash_bucket(
        'marital_status', hash_bucket_size=100),
    layers.sparse_column_with_hash_bucket(
        'relationship', hash_bucket_size=100),
    layers.sparse_column_with_hash_bucket(
        'workclass', hash_bucket_size=100),
    layers.sparse_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000),
    layers.sparse_column_with_hash_bucket(
        'native_country', hash_bucket_size=1000),

    # Continuous base columns.
    layers.real_valued_column('age'),
    layers.real_valued_column('education_num'),
    layers.real_valued_column('capital_gain'),
    layers.real_valued_column('capital_loss'),
    layers.real_valued_column('hours_per_week'),
]


def build_estimator(model_dir, embedding_size=8, hidden_units=None):
  (gender, race, education, marital_status, relationship,
   workclass, occupation, native_country, age,
   education_num, capital_gain, capital_loss, hours_per_week) = INPUT_COLUMNS
  """Build an estimator."""
  # Sparse base columns.
  # Reused Transformations.
  age_buckets = layers.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  wide_columns = [
      layers.crossed_column(
          [education, occupation], hash_bucket_size=int(1e4)),
      layers.crossed_column(
          [age_buckets, race, occupation], hash_bucket_size=int(1e6)),
      layers.crossed_column(
          [native_country, occupation], hash_bucket_size=int(1e4)),
      gender,
      native_country,
      education,
      occupation,
      workclass,
      marital_status,
      relationship,
      age_buckets,
  ]

  deep_columns = [
      layers.embedding_column(workclass, dimension=embedding_size),
      layers.embedding_column(education, dimension=embedding_size),
      layers.embedding_column(marital_status, dimension=embedding_size),
      layers.embedding_column(gender, dimension=embedding_size),
      layers.embedding_column(relationship, dimension=embedding_size),
      layers.embedding_column(race, dimension=embedding_size),
      layers.embedding_column(native_country, dimension=embedding_size),
      layers.embedding_column(occupation, dimension=embedding_size),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]

  return tf.contrib.learn.DNNLinearCombinedClassifier(
      model_dir=model_dir,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25])


def is_sparse(column):
  return isinstance(column, layers.feature_column._SparseColumn)


def feature_columns_to_placeholders(feature_columns, default_batch_size=None):
    return {
        column.name: tf.placeholder(
            tf.string if is_sparse(column) else tf.float32,
            [default_batch_size]
        )
        for column in feature_columns
    }


def serving_input_fn():
    feature_placeholders = feature_columns_to_placeholders(INPUT_COLUMNS)
    features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholders.items()
    }
    return input_fn_utils.InputFnOps(
      features,
      None,
      feature_placeholders
    )


def generate_input_fn(filename, num_epochs=None, batch_size=40):
  def _input_fn():
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)
    reader = tf.TextLineReader()
    _, value = reader.read_up_to(filename_queue, num_records=batch_size)
    value_column = tf.expand_dims(value, -1)

    columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))

    # remove the fnlwgt key, which is not used
    features.pop('fnlwgt', None)
    income_int = tf.to_int32(tf.equal(features.pop(LABEL_COLUMN), ' >50K'))
    return features, income_int
  return _input_fn
