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

"""Implements the vanilla tensorflow model on single node."""

# See https://goo.gl/JZ6hlH to contrast this with DNN combined
# which the high level estimator based sample implements.
import multiprocessing

import tensorflow as tf
from tensorflow.python.ops import string_ops

# See tutorial on wide and deep
# https://www.tensorflow.org/tutorials/wide_and_deep/
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/feature_column.py

# csv columns in the input file
CSV_COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race',
               'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'income_bracket')

CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''],
                       [''], [0], [0], [0], [''], ['']]

# Categorical columns with vocab size
CATEGORICAL_COLS = (('education', 16), ('marital_status', 7),
                    ('relationship', 6), ('workclass', 9), ('occupation', 15),
                    ('native_country', 42), ('gender', [' Male', ' Female']),
                    ('race', 5))

CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss',
                   'hours_per_week')

LABELS = [' <=50K', ' >50K']
LABEL_COLUMN = 'income_bracket'

UNUSED_COLUMNS = set(CSV_COLUMNS) - set(
    zip(*CATEGORICAL_COLS)[0] + CONTINUOUS_COLS + (LABEL_COLUMN,))

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
CSV, EXAMPLE, JSON = 'CSV', 'EXAMPLE', 'JSON'
PREDICTION_MODES = [CSV, EXAMPLE, JSON]


def model_fn(mode,
             features,
             labels,
             embedding_size=8,
             hidden_units=[100, 70, 50, 20],
             learning_rate=0.1):
  """Create a Feed forward network classification network

  Args:
    mode (string): Mode running training, evaluation or prediction
    features (dict): Dictionary of input feature Tensors
    labels (Tensor): Class label Tensor
    hidden_units (list): Hidden units
    learning_rate (float): Learning rate for the SGD

  Returns:
    Depending on the mode returns Tuple or Dict
  """
  label_values = tf.constant(LABELS)

  # Keep variance constant with changing embedding sizes.
  embed_initializer = tf.truncated_normal_initializer(
      stddev=(1.0 / tf.sqrt(float(embedding_size))))

  with tf.variable_scope('embeddings', initializer=embed_initializer):
    # Convert categorical (string) values to embeddings
    for col, vals in CATEGORICAL_COLS:
      bucket_size = vals if isinstance(vals, int) else len(vals)
      embeddings = tf.get_variable(
          col,
          shape=[bucket_size, embedding_size]
      )

      if isinstance(vals, int):
        indices = string_ops.string_to_hash_bucket_fast(
          features[col], bucket_size)
      else:
        table = tf.contrib.lookup.index_table_from_tensor(vals)
        indices = table.lookup(features[col])

      features[col] = tf.nn.embedding_lookup(embeddings, indices)

  for col in CONTINUOUS_COLS:
    # Give continuous columns an extra trivial dimension
    # So they can be concatenated with embedding tensors
    features[col] = tf.expand_dims(tf.to_float(features[col]), -1)

  # Concatenate the (now all dense) features.
  # We need to sort the tensors so that they end up in the same order for
  # prediction, evaluation, and training
  sorted_feature_tensors = zip(*sorted(features.iteritems()))[1]
  inputs = tf.concat(sorted_feature_tensors, 1)

  # Build the DNN
  curr_layer = inputs

  for layer_size in hidden_units:
    curr_layer = tf.layers.dense(
        curr_layer,
        layer_size,
        activation=tf.nn.relu,
        # This initializer prevents variance from exploding or vanishing when
        # compounded through different sized layers.
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    )

  # Add the output layer
  logits = tf.layers.dense(
    curr_layer,
    len(LABELS),
    # Do not use ReLU on last layer
    activation=None,
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
  )

  if mode in (PREDICT, EVAL):
    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)

  if mode in (TRAIN, EVAL):
    # Convert the string label column to indices
    # Build a lookup table inside the graph
    table = tf.contrib.lookup.index_table_from_tensor(label_values)

    # Use the lookup table to convert string labels to ints
    label_indices = table.lookup(labels)

    # Make labels a vector
    label_indices_vector = tf.squeeze(label_indices)

    # global_step is necessary in eval to correctly load the step
    # of the checkpoint we are evaluating
    global_step = tf.contrib.framework.get_or_create_global_step()

  if mode == PREDICT:
    # Convert predicted_indices back into strings
    return {
        'predictions': tf.gather(label_values, predicted_indices),
        'confidence': tf.reduce_max(probabilities, axis=1)
    }

  if mode == TRAIN:
    # Build training operation.
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label_indices_vector))
    tf.summary.scalar('loss', cross_entropy)
    train_op = tf.train.FtrlOptimizer(
        learning_rate=learning_rate,
        l1_regularization_strength=3.0,
        l2_regularization_strength=10.0
    ).minimize(cross_entropy, global_step=global_step)
    return train_op, global_step

  if mode == EVAL:
    # Return accuracy and area under ROC curve metrics
    # See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    # See https://www.kaggle.com/wiki/AreaUnderCurve\
    labels_one_hot = tf.one_hot(
        label_indices_vector,
        depth=label_values.shape[0],
        on_value=True,
        off_value=False,
        dtype=tf.bool
    )
    return {
        'accuracy': tf.metrics.accuracy(label_indices, predicted_indices),
        'auroc': tf.metrics.auc(labels_one_hot, probabilities)
    }


def csv_serving_input_fn(default_batch_size=None):
  """Build the serving inputs.

  Args:
    default_batch_size (int): Batch size for the tf.placeholder shape
  """
  csv_row = tf.placeholder(
      shape=[default_batch_size],
      dtype=tf.string
  )
  features = parse_csv(csv_row)
  features.pop(LABEL_COLUMN)
  return features, {'csv_row': csv_row}


def example_serving_input_fn(default_batch_size=None):
  """Build the serving inputs.

  Args:
    default_batch_size (int): Batch size for the tf.placeholder shape
  """
  feature_spec = {}
  for feat in CONTINUOUS_COLS:
    feature_spec[feat] = tf.FixedLenFeature(shape=[], dtype=tf.int64)

  for feat, _ in CATEGORICAL_COLS:
    feature_spec[feat] = tf.FixedLenFeature(shape=[], dtype=tf.string)

  example_bytestring = tf.placeholder(
      shape=[default_batch_size],
      dtype=tf.string,
  )
  features = tf.parse_example(example_bytestring, feature_spec)
  return features, {'example': example_bytestring}


def json_serving_input_fn(default_batch_size=None):
  """Build the serving inputs.

  Args:
    default_batch_size (int): Batch size for the tf.placeholder shape
  """
  inputs = {}
  for feat in CONTINUOUS_COLS:
    inputs[feat] = tf.placeholder(
        shape=[default_batch_size], dtype=tf.float32)

  for feat, _ in CATEGORICAL_COLS:
    inputs[feat] = tf.placeholder(
        shape=[default_batch_size], dtype=tf.string)
  return inputs, inputs


SERVING_INPUT_FUNCTIONS = {
    JSON: json_serving_input_fn,
    CSV: csv_serving_input_fn,
    EXAMPLE: example_serving_input_fn
}


def parse_csv(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""

  # Takes a rank-1 tensor and converts it into rank-2 tensor
  # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
  # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
  # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
  columns = tf.decode_csv(
      rows_string_tensor, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)
  return features


def input_fn(filenames,
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
  return features, features.pop(LABEL_COLUMN)
