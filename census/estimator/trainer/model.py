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


# Define the format of your input data including unused columns
CSV_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'gender',
               'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'income_bracket']
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
    tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', [' Female', ' Male']),

    tf.feature_column.categorical_column_with_vocabulary_list(
        'race',
        [' Amer-Indian-Eskimo', ' Asian-Pac-Islander',
         ' Black', ' Other', ' White']
    ),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'education',
        [' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
         ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',
         ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th',
         ' 1st-4th', ' Preschool', ' 12th']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status',
        [' Married-civ-spouse', ' Divorced', ' Married-spouse-absent',
         ' Never-married', ' Separated', ' Married-AF-spouse', ' Widowed']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',
        [' Husband', ' Not-in-family', ' Wife', ' Own-child', ' Unmarried',
         ' Other-relative']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass',
        [' Self-emp-not-inc', ' Private', ' State-gov',
         ' Federal-gov', ' Local-gov', ' ?', ' Self-emp-inc',
         ' Without-pay', ' Never-worked']
    ),

    # For columns with a large number of values, or unknown values
    # We can use a hash function to convert to categories.
    tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=100, dtype=tf.string),
    tf.feature_column.categorical_column_with_hash_bucket(
        'native_country', hash_bucket_size=100, dtype=tf.string),

    # Continuous base columns.
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('education_num'),
    tf.feature_column.numeric_column('capital_gain'),
    tf.feature_column.numeric_column('capital_loss'),
    tf.feature_column.numeric_column('hours_per_week'),
]

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - \
    {LABEL_COLUMN}


def build_estimator(config, embedding_size=8, hidden_units=None):
  """Build a wide and deep model for predicting income category.

  Wide and deep models use deep neural nets to learn high level abstractions
  about complex features or interactions between such features.
  These models then combined the outputs from the DNN with a linear regression
  performed on simpler features. This provides a balance between power and
  speed that is effective on many structured data problems.

  You can read more about wide and deep models here:
  https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html

  To define model we can use the prebuilt DNNCombinedLinearClassifier class,
  and need only define the data transformations particular to our dataset, and
  then
  assign these (potentially) transformed features to either the DNN, or linear
  regression portion of the model.

  Args:
    config: tf.contrib.learn.RunConfig defining the runtime environment for the
      estimator (including model_dir).
    embedding_size: int, the number of dimensions used to represent categorical
      features when providing them as inputs to the DNN.
    hidden_units: [int], the layer sizes of the DNN (input layer first)
    learning_rate: float, the learning rate for the optimizer.
  Returns:
    A DNNCombinedLinearClassifier
  """
  (gender, race, education, marital_status, relationship,
   workclass, occupation, native_country, age,
   education_num, capital_gain, capital_loss, hours_per_week) = INPUT_COLUMNS
  """Build an estimator."""

  # Reused Transformations.
  # Continuous columns can be converted to categorical via bucketization
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  wide_columns = [
      # Interactions between different categorical features can also
      # be added as new virtual features.
      tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=int(1e4)),
      tf.feature_column.crossed_column(
          [age_buckets, race, 'occupation'], hash_bucket_size=int(1e6)),
      tf.feature_column.crossed_column(
          ['native_country', 'occupation'], hash_bucket_size=int(1e4)),
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
      # Use indicator columns for low dimensional vocabularies
      tf.feature_column.indicator_column(workclass),
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(gender),
      tf.feature_column.indicator_column(relationship),
      tf.feature_column.indicator_column(race),

      # Use embedding columns for high dimensional vocabularies
      tf.feature_column.embedding_column(
          native_country, dimension=embedding_size),
      tf.feature_column.embedding_column(occupation, dimension=embedding_size),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]

  return tf.contrib.learn.DNNLinearCombinedClassifier(
      config=config,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25],
      fix_global_step_increment_bug=True
  )


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
  table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))

  # Use the hash table to convert string labels to ints and one-hot encode
  return table.lookup(label_string_tensor)


# ************************************************************************
# YOU NEED NOT MODIFY ANYTHING BELOW HERE TO ADAPT THIS MODEL TO YOUR DATA
# ************************************************************************


def csv_serving_input_fn():
  """Build the serving inputs."""
  csv_row = tf.placeholder(
      shape=[None],
      dtype=tf.string
  )
  features = parse_csv(csv_row)
  features.pop(LABEL_COLUMN)
  return tf.contrib.learn.InputFnOps(features, None, {'csv_row': csv_row})


def example_serving_input_fn():
  """Build the serving inputs."""
  example_bytestring = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  feature_scalars = tf.parse_example(
      example_bytestring,
      tf.feature_column.make_parse_example_spec(INPUT_COLUMNS)
  )
  features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_scalars.iteritems()
  }
  return tf.contrib.learn.InputFnOps(
      features,
      None,  # labels
      {'example_proto': example_bytestring}
  )

# [START serving-function]
def json_serving_input_fn():
  """Build the serving inputs."""
  inputs = {}
  for feat in INPUT_COLUMNS:
    inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

  features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in inputs.iteritems()
  }
  return tf.contrib.learn.InputFnOps(features, None, inputs)
# [END serving-function]

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}


def parse_csv(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""

  # Takes a rank-1 tensor and converts it into rank-2 tensor
  # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
  # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
  # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)
  return features


def generate_input_fn(filenames,
                      num_epochs=None,
                      shuffle=True,
                      skip_header_lines=0,
                      batch_size=200):
  """Generates an input function for training or evaluation.
  This uses the input pipeline based approach using file name queue
  to read data so that entire data is not loaded in memory.

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

  # Parse the CSV File
  features = parse_csv(rows)

  # This operation builds up a buffer of parsed tensors, so that parsing
  # input data doesn't block training
  # If requested it will also shuffle
  if shuffle:
    features = tf.train.shuffle_batch(
        features,
        batch_size,
        min_after_dequeue=2 * batch_size + 1,
        capacity=batch_size * 10,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
        allow_smaller_final_batch=True
    )
  else:
    features = tf.train.batch(
        features,
        batch_size,
        capacity=batch_size * 10,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
        allow_smaller_final_batch=True
    )

  return features, parse_label_column(features.pop(LABEL_COLUMN))
