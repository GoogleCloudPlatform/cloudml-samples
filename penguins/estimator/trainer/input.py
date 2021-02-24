# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Defines a Wide + Deep model for classification on structured data.

Tutorial on wide and deep: https://www.tensorflow.org/tutorials/wide_and_deep/
"""

import multiprocessing
import tensorflow as tf

from constants import constants
import trainer.featurizer as featurizer


def _decode_csv(line):
    """Takes the string input tensor and returns a dict of rank-2 tensors."""

    # Takes a rank-1 tensor and converts it into rank-2 tensor
    # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
    # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
    # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
    row_columns = tf.expand_dims(line, -1)
    columns = tf.decode_csv(
        row_columns, record_defaults=constants.CSV_COLUMN_DEFAULTS)
    features = dict(zip(constants.CSV_COLUMNS, columns))

    # Remove unused columns
    unused_columns = set(constants.CSV_COLUMNS) - {col.name for col in
                                                   featurizer.INPUT_COLUMNS} - {
                         constants.LABEL_COLUMN}
    for col in unused_columns:
        features.pop(col)
    return features


def _parse_label_column(label_string_tensor):
    """Parses a string tensor into the label tensor.

    Args:
      label_string_tensor: Tensor of dtype string. Result of parsing the CSV
        column specified by LABEL_COLUMN.

    Returns:
      A Tensor of the same shape as label_string_tensor, should return
      an int64 Tensor representing the label index for classification tasks,
      and a float32 Tensor representing the value for a regression task.
    """
    # Build a Hash Table inside the graph
    table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(constants.LABELS))

    # Use the hash table to convert string labels to ints and one-hot encode
    return table.lookup(label_string_tensor)


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=200,
             num_parallel_calls=None,
             prefetch_buffer_size=None):
    """Generates features and labels for training or evaluation.

    This uses the input pipeline based approach using file name queue
    to read data so that entire data is not loaded in memory.

    Args:
        filenames: [str] A List of CSV file(s) to read data from.
        num_epochs: (int) how many times through to read the data. If None will
          loop through data indefinitely
        shuffle: (bool) whether or not to randomize the order of data. Controls
          randomization of both file order and line order within files.
        skip_header_lines: (int) set to non-zero in order to skip header
        lines in CSV files.
        batch_size: (int) First dimension size of the Tensors returned by
        input_fn

    Returns:
        A (features, indices) tuple where features is a dictionary of
          Tensors, and indices is a single Tensor of label indices.
    """
    if num_parallel_calls is None:
        num_parallel_calls = multiprocessing.cpu_count()

    if prefetch_buffer_size is None:
        prefetch_buffer_size = 1024

    dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(
        _decode_csv, num_parallel_calls).prefetch(prefetch_buffer_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    iterator = dataset.repeat(num_epochs).batch(
        batch_size).make_one_shot_iterator()
    features = iterator.get_next()
    return features, _parse_label_column(features.pop(constants.LABEL_COLUMN))


# ************************************************************************
# YOU NEED NOT MODIFY ANYTHING BELOW HERE TO ADAPT THIS MODEL TO YOUR DATA
# ************************************************************************


def csv_serving_input_fn():
    """Build the serving inputs."""
    csv_row = tf.placeholder(shape=[None], dtype=tf.string)
    features = _decode_csv(csv_row)
    features.pop(constants.LABEL_COLUMN)
    return tf.estimator.export.ServingInputReceiver(features,
                                                    {'csv_row': csv_row})


def example_serving_input_fn():
    """Build the serving inputs."""
    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    features = tf.parse_example(
        example_bytestring,
        tf.feature_column.make_parse_example_spec(featurizer.INPUT_COLUMNS))
    return tf.estimator.export.ServingInputReceiver(
        features, {'example_proto': example_bytestring})


# [START serving-function]
def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in featurizer.INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


# [END serving-function]

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}
