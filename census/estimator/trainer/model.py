# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Define a Wide + Deep model for classification on structured data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils


# Define the format of your input data including unused columns
CSV_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'gender',
               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
               'income_bracket']
CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                       [0], [0], [0], [''], ['']]
LABEL_COLUMN = 'income_bracket'
LABELS = [' <=50K', ' >50K']

# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
    # Categorical base columns

    # For categorical columns with known values we can provide lists
    # of values ahead of time.
    layers.sparse_column_with_keys(column_name='gender', keys=['female', 'male']),

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

    # Otherwise we can use a hashing function to bucket the categories
    layers.sparse_column_with_hash_bucket('education', hash_bucket_size=1000),
    layers.sparse_column_with_hash_bucket('marital_status', hash_bucket_size=100),
    layers.sparse_column_with_hash_bucket('relationship', hash_bucket_size=100),
    layers.sparse_column_with_hash_bucket('workclass', hash_bucket_size=100),
    layers.sparse_column_with_hash_bucket('occupation', hash_bucket_size=1000),
    layers.sparse_column_with_hash_bucket('native_country', hash_bucket_size=1000),

    # Continuous base columns.
    layers.real_valued_column('age'),
    layers.real_valued_column('education_num'),
    layers.real_valued_column('capital_gain'),
    layers.real_valued_column('capital_loss'),
    layers.real_valued_column('hours_per_week'),
]

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - {LABEL_COLUMN}

def build_estimator(model_dir, embedding_size=8, hidden_units=None):
  """Build a wide and deep model for predicting income category.

  Wide and deep models use deep neural nets to learn high level abstractions
  about complex features or interactions between such features.
  These models then combined the outputs from the DNN with a linear regression
  performed on simpler features. This provides a balance between power and
  speed that is effective on many structured data problems.

  You can read more about wide and deep models here:
  https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html

  To define model we can use the prebuilt DNNCombinedLinearClassifier class,
  and need only define the data transformations particular to our dataset, and then
  assign these (potentially) transformed features to either the DNN, or linear
  regression portion of the model.

  Args:
    model_dir: str, the model directory used by the Classifier for checkpoints
      summaries and exports.
    embedding_size: int, the number of dimensions used to represent categorical
      features when providing them as inputs to the DNN.
    hidden_units: [int], the layer sizes of the DNN (input layer first)
  Returns:
    A DNNCombinedLinearClassifier
  """
  (gender, race, education, marital_status, relationship,
   workclass, occupation, native_country, age,
   education_num, capital_gain, capital_loss, hours_per_week) = INPUT_COLUMNS
  """Build an estimator."""

  # Reused Transformations.
  # Continuous columns can be converted to categorical via bucketization
  age_buckets = layers.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  wide_columns = [
      # Interactions between different categorical features can also
      # be added as new virtual features.
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


# ************************************************************************
# YOU NEED NOT MODIFY ANYTHING BELOW HERE TO ADAPT THIS MODEL TO YOUR DATA
# ************************************************************************


def serving_input_fn():
  """Builds the input subgraph for prediction.

  This serving_input_fn accepts raw Tensors inputs which will be fed to the
  server as JSON dictionaries. The values in the JSON dictionary will be
  converted to Tensors of the appropriate type.

  Returns:
     tf.contrib.learn.input_fn_utils.InputFnOps, a named tuple
     (features, labels, inputs) where features is a dict of features to be
     passed to the Estimator, labels is always None for prediction, and
     inputs is a dictionary of inputs that the prediction server should expect
     from the user.
  """
  feature_placeholders = {
      column.name: tf.placeholder(column.dtype, [None])
      for column in INPUT_COLUMNS
  }
  # DNNCombinedLinearClassifier expects rank 2 Tensors, but inputs should be
  # rank 1, so that we can provide scalars to the server
  features = {
    key: tf.expand_dims(tensor, -1)
    for key, tensor in feature_placeholders.items()
  }
  return input_fn_utils.InputFnOps(
    features,
    None,
    feature_placeholders
  )


def generate_input_fn(filenames,
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
  def _input_fn():
    files = tf.concat([
      tf.train.match_filenames_once(filename)
      for filename in filenames
    ], axis=0)

    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=shuffle)
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
      # This operation maintains a buffer of Tensors so that inputs are
      # well shuffled even between batches.
      features = tf.train.shuffle_batch(
          features,
          batch_size,
          capacity=batch_size * 10,
          min_after_dequeue=batch_size*2 + 1,
          num_threads=multiprocessing.cpu_count(),
          enqueue_many=True,
          allow_smaller_final_batch=True
      )
    label_tensor = parse_label_column(features.pop(LABEL_COLUMN))
    return features, label_tensor
  return _input_fn
