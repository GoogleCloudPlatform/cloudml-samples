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

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'


def model_fn(mode,
             features,
             labels,
             hidden_units=[100, 70, 50, 20],
             learning_rate=0.1):
  """Create a Feed forward network classification network

  Args:
    input_x (tf.placeholder): Feature placeholder input
    hidden_units (list): Hidden units
    num_classes (int): Number of classes

  Returns:
    Tuple (train_op, accuracy_op, global_step, predictions): Tuple containing
    training graph, accuracy graph, global step and predictions
  """
  label_values = tf.constant(LABELS)

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

  if mode in (PREDICT, EVAL):
    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)


  if mode in (TRAIN, EVAL):
    # Conver the string label column to indices
    # Build a Hash Table inside the graph
    table = tf.contrib.lookup.string_to_index_table_from_tensor(
        label_values)
    # Use the hash table to convert string labels to ints
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
        'confidence': tf.gather(probabilities, predicted_indices)
    }

  if mode == TRAIN:
    # Build training operation.
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label_indices_vector))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        cross_entropy, global_step=global_step)
    return train_op, global_step

  if mode == EVAL:
    return {
        'accuracy': tf.contrib.metrics.streaming_accuracy(
            predicted_indices, label_indices),
        'auroc': tf.contrib.metrics.streaming_auc(predicted_indices, label_indices)
    }


def build_serving_inputs(mode, default_batch_size=None):
  if mode == 'CSV':
    placeholders = {'csv_row': tf.placeholder(
        shape=[default_batch_size],
        dtype=tf.string
    )}
    features = parse_csv(placeholders['csv_row'])
    features.pop(LABEL_COLUMN)
  else:
    feature_spec = {}
    for feat in CONTINUOUS_COLS:
      feature_spec[feat] = tf.FixedLenFeature(shape=[], dtype=tf.float32)

    for feat, _ in CATEGORICAL_COLS:
      feature_spec[feat] = tf.FixedLenFeature(shape=[], dtype=tf.string)

    tf_record = tf.placeholder(
        shape=[default_batch_size],
        dtype=tf.string,
        name='tf_record'
    )
    feature_scalars = tf.parse_example(tf_record, feature_spec)
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_scalars.iteritems()
    }
    if mode == 'TF_RECORD':
      placeholders = {'tf_record': tf_record}
    else:
      placeholders = feature_scalars

  return features, placeholders


def parse_csv(rows_string_tensor):
  # model_fn expects rank 2 tensors.
  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)
  return features

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

  features = parse_csv(rows)

  # Parse the CSV File
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
        allow_smaller_final_batch=True
    )
  return features, features.pop(LABEL_COLUMN)
