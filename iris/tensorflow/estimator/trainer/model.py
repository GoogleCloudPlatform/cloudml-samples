# Copyright 2018 Google Inc. All Rights Reserved. Licensed under the Apache
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

# The CSV features in our training & test data.
column_names = [
    'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'
]

try:
	xrange  # Python 2
except NameError:
	xrange = range  # Python 3


def get_feature_columns():
  # All our inputs are feature columns of type numeric_column.
  return [
      tf.feature_column.numeric_column(feature, shape=(1,))
      for feature in column_names[:4]
  ]


def build_estimator(config,
                    hidden_units=None,
                    learning_rate=1e-4,
                    num_classes=3):
  """Creates a Deep Neural Network estimator for Multi-class classification.

  By default builds a 3 layer DNN with 10, 20, 10 units respectively.

  Args:
    config: (tf.estimator.RunConfig) Configuration options to save model.
    hidden_units: [int] DNN structure. Example: [10, 20, 10]
    learning_rate: (float) Learning rate.
    num_classes: (int) Number of classes in label.

  Returns:
    A tf.estimator.DNNClassifier object.
  """
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  return tf.estimator.DNNClassifier(
      optimizer=optimizer,
      config=config,
      feature_columns=get_feature_columns(),
      hidden_units=hidden_units,
      n_classes=num_classes)


def _make_input_parser(with_target=True):
  """Returns a parser func according to file_type, task_type and target.

  Need to set record_default for last column to integer instead of float in
  case of classification tasks.

  Args:
    with_target (boolean): Pass label or not.

  Returns:
    It returns a parser.
  """

  def _decode_csv(line):
    """Takes the string input tensor and parses it to feature dict and target.

    All the columns except the first one are treated as feature column. The
    first column is expected to be the target.
    Only returns target for if with_target is True.

    Args:
      line: csv rows in tensor format.

    Returns:
      features: A dictionary of features with key as "column_names" from
        self._column_header.
      target: tensor of target values which is the first column of the file.
        This will only be returned if with_target==True.
    """
    column_header = column_names if with_target else column_names[:4]
    record_defaults = [[0.] for _ in xrange(len(column_names) - 1)]
    # Pass label as integer.
    if with_target:
      record_defaults.append([0])
    columns = tf.decode_csv(line, record_defaults=record_defaults)
    features = dict(zip(column_header, columns))
    target = features.pop(column_names[4]) if with_target else None
    return features, target

  return _decode_csv


def serving_input_receiver_fn():
  """This is used to define inputs to serve the model.

  Returns:
    A ServingInputReciever object.
  """
  csv_row = tf.placeholder(shape=[None], dtype=tf.string)
  features, _ = _make_input_parser(with_target=False)(csv_row)
  return tf.estimator.export.ServingInputReceiver(features,
                                                  {'csv_row': csv_row})


def input_fn(filename, batch_size=32, shuffle=False):
  """Creates an input function reading a file using the Dataset API.

  Args:
    filename: (str) The data file to read.
    batch_size: (int) The size of training examples in one iteration.
    shuffle: (boolean) Whether the record order should be randomized.

  Returns:
    Features and Labels for processing.
  """

  def _input_fn():
    """The input_fn."""
    dataset = (
        tf.data.TextLineDataset(filename).skip(
            1)  # Skip the first line (which does not have data)
        .map(_make_input_parser(with_target=True))
    )  # Transform each elem _make_input_parser.

    if shuffle:
      dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat().batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  return _input_fn()
