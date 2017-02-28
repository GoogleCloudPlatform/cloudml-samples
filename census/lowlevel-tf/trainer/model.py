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

# See tutorial on wide and deep https://www.tensorflow.org/tutorials/wide_and_deep/
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/feature_column.py

# csv columns in the input file
CSV_COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race',
               'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'income_bracket')

CSV_COLUMN_DEFAULTS = [[0.], [''], [0.], [''], [0.], [''], [''], [''], [''],
                       [''], [0.], [0.], [0.], [''], ['']]

# Categorical columns with vocab size
HASH_BUCKET_COLS = (('education', 16), ('marital_status', 7),
                    ('relationship', 6), ('workclass', 9), ('occupation', 15),
                    ('native_country', 42))
KEY_COLS = (('gender', ('female', 'male')), ('race', ('Amer-Indian-Eskimo',
                                                      'Asian-Pac-Islander',
                                                      'Black',
                                                      'Other',
                                                      'White')))


CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss',
                   'hours_per_week')

CATEGORICAL_COLS = HASH_BUCKET_COLS + tuple((col, len(keys)) for col, keys in KEY_COLS)
LABELS = [' <=50K', ' >50K']
LABEL_COLUMN = 'income_bracket'

UNUSED_COLUMNS = set(CSV_COLUMNS) - set(
    zip(*CATEGORICAL_COLS)[0] + CONTINUOUS_COLS + (LABEL_COLUMN,))


# Graph creation section for training and evaluation
def model_fn(features,
             labels,
             hidden_units=[100, 70, 50, 20],
             learning_rate=0.5,
             batch_size=40):
  """Create a Feed forward network classification network

  Args:
    input_x (tf.placeholder): Feature placeholder input
    hidden_units (list): Hidden units
    num_classes (int): Number of classes

  Returns:

  """
  # Convert categorical (string) values to one_hot values
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

  # Concatenate the (now all dense) features.
  inputs = tf.concat(features.values(), 1)

  # Build the DNN

  layers_size = [inputs.get_shape()[1]] + hidden_units
  layers_shape = zip(layers_size[0:], layers_size[1:] + [len(LABELS)])

  curr_layer = inputs
  with tf.variable_scope('dnn',
                         initializer=tf.truncated_normal_initializer()):
    # Creates the relu hidden layers
    for num, shape in enumerate(layers_shape):
      with tf.variable_scope('relu_{}'.format(num)):

        weights = tf.get_variable('weights', shape)

        biases = tf.get_variable(
            'biases', shape[1], initializer=tf.zeros_initializer(tf.float32))

      curr_layer = tf.nn.relu(tf.matmul(curr_layer, weights) + biases)

  # Make predictions
  logits = curr_layer
  probabilities = tf.nn.softmax(logits)
  predictions = tf.argmax(probabilities, 1)

  # Make labels a vector
  labels = tf.squeeze(labels)

  # Build training operation.
  global_step = tf.contrib.framework.get_or_create_global_step()
  cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        cross_entropy, global_step=global_step)

  accuracy_op = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))

  return train_op, accuracy_op, global_step, predictions


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

  # model_fn expects rank 2 tensors.
  row_columns = tf.expand_dims(rows, -1)

  # Parse the CSV File
  columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)

  if shuffle:
    # This operation builds up a buffer of rows so that, even between batches,
    # rows are fed to training in a suitably randomized order.
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
