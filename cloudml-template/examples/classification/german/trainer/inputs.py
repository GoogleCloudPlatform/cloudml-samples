#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import logging
import tensorflow as tf
import tensorflow_model_analysis as tfma
import metadata

# ******************************************************************************
# YOU NEED NOT TO CHANGE THESE HELPER FUNCTIONS
# *****************************************************************************


def get_feature_spec(is_serving=False):
  """Create feature_spec from metadata. Used for parsing tf examples.
  Args:
    is_serving: boolean - whether to create feature_spec for training o serving.
  Returns:
    feature_spec
  """

  column_names = metadata.SERVING_COLUMN_NAMES \
    if is_serving else metadata.COLUMN_NAMES

  feature_spec = {}

  for feature_name in column_names:
    if feature_name in metadata.NUMERIC_FEATURE_NAMES_WITH_STATS:
      feature_spec[feature_name] = tf.FixedLenFeature(shape=1, dtype=tf.float32)
    elif feature_name in metadata.CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY:
      feature_spec[feature_name] = tf.FixedLenFeature(shape=1, dtype=tf.int32)
    elif feature_name in metadata.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY:
      feature_spec[feature_name] = tf.FixedLenFeature(shape=1, dtype=tf.string)
    elif feature_name in metadata.CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET:
      feature_spec[feature_name] = tf.FixedLenFeature(shape=1, dtype=tf.string)
    elif feature_name == metadata.TARGET_NAME:
      if metadata.TASK_TYPE == 'classification':
        feature_spec[feature_name] = tf.FixedLenFeature(shape=1,
                                                        dtype=tf.string)
      else:
        feature_spec[feature_name] = tf.FixedLenFeature(shape=1,
                                                        dtype=tf.float32)

  return feature_spec


def parse_csv(csv_row, is_serving=False):
  """Takes the string input tensor (csv) and returns a dict of rank-2 tensors.

  Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to
  its data type (inferred from the metadata).

  Args:
    csv_row: rank-2 tensor of type string (csv).
    is_serving: boolean to indicate whether this function is called during
      serving or training, since the csv_row serving input is different than
      the training input (i.e., no target column).
  Returns:
    rank-2 tensor of the correct data type.
  """
  if is_serving:
    column_names = metadata.SERVING_COLUMN_NAMES
    defaults = []
    # create the defaults for the serving columns.
    for serving_feature in metadata.SERVING_COLUMN_NAMES:
      feature_index = metadata.COLUMN_NAMES.index(serving_feature)
      defaults.append(metadata.DEFAULTS[feature_index])
  else:
    column_names = metadata.COLUMN_NAMES
    defaults = metadata.DEFAULTS

  columns = tf.decode_csv(csv_row, record_defaults=defaults)
  features = dict(zip(column_names, columns))

  return features


# ******************************************************************************
# YOU MAY IMPLEMENT THIS FUNCTION FOR CUSTOM FEATURE ENGINEERING
# ******************************************************************************


def process_features(features):
  """ Use to implement custom feature engineering logic.

  Default behaviour is to return the original feature tensors dictionary as-is.

  Args:
      features: {string:tensors} - dictionary of feature tensors
  Returns:
      {string:tensors}: extended feature tensors dictionary
  """

  # examples - given:
  # 'x' and 'y' are two numeric features:
  # 'alpha' and 'beta' are two categorical features

  # # create new features using custom logic
  # features['x_2'] = tf.pow(features['x'],2)
  # features['y_2'] = tf.pow(features['y'], 2)
  # features['xy'] = features['x'] * features['y']
  # features['sin_x'] = tf.sin(features['x'])
  # features['cos_y'] = tf.cos(features['x'])
  # features['log_xy'] = tf.log(features['xy'])
  # features['sqrt_xy'] = tf.sqrt(features['xy'])

  # # add created features to metadata (if not already defined in metadata.py)
  # NUMERIC_FEATURE_NAMES_WITH_STATS['x_2']: None
  # NUMERIC_FEATURE_NAMES_WITH_STATS['y_2']: None
  # ....


  return features


# ******************************************************************************
# YOU NEED NOT TO CHANGE THIS FUNCTION TO READ DATA FILES
# ******************************************************************************


def make_input_fn(file_pattern,
                  file_encoding='csv',
                  mode=tf.estimator.ModeKeys.EVAL,
                  has_header=False,
                  batch_size=200,
                  multi_threading=True):
  """Makes an input function for reading training and evaluation data file(s).

  Args:
      file_pattern: str - file name or file name patterns from which to read the data.
      mode: tf.estimator.ModeKeys - either TRAIN or EVAL.
          Used to determine whether or not to randomize the order of data.
      file_encoding: type of the text files. Can be 'csv' or 'tfrecords'
      has_header: boolean - set to non-zero in order to skip header lines in CSV files.
      num_epochs: int - how many times through to read the data.
        If None will loop through data indefinitely
      batch_size: int - first dimension size of the Tensors returned by input_fn
      multi_threading: boolean - indicator to use multi-threading or not
  Returns:
      A function () -> (features, indices) where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
  """

  shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
  num_epochs = None if mode == tf.estimator.ModeKeys.TRAIN else 1
  num_threads = multiprocessing.cpu_count() if multi_threading else 1
  buffer_size = 2 * batch_size + 1

  logging.info("Making input_fn...")
  logging.info("Mode: {}.".format(mode))
  logging.info("Input file(s): {}.".format(file_pattern))
  logging.info("Files encoding: {}.".format(file_encoding))
  logging.info("Batch size: {}.".format(batch_size))
  logging.info("Epoch count: {}.".format(num_epochs))
  logging.info("Thread count: {}.".format(num_threads))
  logging.info("Shuffle: {}.".format(shuffle))

  def _input_fn():
    if file_encoding == 'csv':
      dataset = tf.data.experimental.make_csv_dataset(
        file_pattern,
        batch_size,
        column_names=metadata.COLUMN_NAMES,
        column_defaults=metadata.DEFAULTS,
        label_name=metadata.TARGET_NAME,
        field_delim='|',
        header=has_header,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_buffer_size=buffer_size,
        num_parallel_reads=num_threads,
        sloppy=True,
      )
    else:
      dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern,
        batch_size,
        features=get_feature_spec(),
        reader=tf.data.TFRecordDataset,
        label_key=metadata.TARGET_NAME,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_buffer_size=buffer_size,
        reader_num_threads=num_threads,
        parser_num_threads=num_threads,
        sloppy_ordering=True,
        drop_final_batch=False
      )

    dataset = dataset.map(
      lambda features, target: (process_features(features), target))

    return dataset

  return _input_fn


# ******************************************************************************
# SERVING INPUT FUNCTIONS - YOU NEED NOT TO CHANGE THE FOLLOWING PART
# ******************************************************************************


def json_serving_input_receiver_fn():
  """Creating an ServingInputReceiver object for JSON data.

  Returns:
    ServingInputReceiver
  """

  # Note that the inputs are raw features, not transformed features.
  receiver_tensors = {}

  for column_name in metadata.SERVING_COLUMN_NAMES:
    if column_name in metadata.CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY:
      receiver_tensors[column_name] = tf.placeholder(
        shape=[None], dtype=tf.int32)
    elif column_name in metadata.NUMERIC_FEATURE_NAMES_WITH_STATS:
      receiver_tensors[column_name] = tf.placeholder(
        shape=[None], dtype=tf.float32)
    else:
      receiver_tensors[column_name] = tf.placeholder(
        shape=[None], dtype=tf.string)

  features = {
    key: tf.expand_dims(tensor, -1)
    for key, tensor in receiver_tensors.items()
  }

  return tf.estimator.export.ServingInputReceiver(
    features=process_features(features),
    receiver_tensors=receiver_tensors
  )


def csv_serving_input_receiver_fn():
  """Creating an ServingInputReceiver object for CSV data.

  Returns:
    ServingInputReceiver
  """

  # Note that the inputs are raw features, not transformed features.
  csv_row = tf.placeholder(shape=[None], dtype=tf.string)
  features = parse_csv(csv_row, is_serving=True)

  return tf.estimator.export.ServingInputReceiver(
    features=process_features(features),
    receiver_tensors={'csv_row': csv_row}
  )


def example_serving_input_receiver_fn():
  """Creating an ServingInputReceiver object for TFRecords data.

  Returns:
    ServingInputReceiver
  """

  # Note that the inputs are raw features, not transformed features.
  receiver_tensors = tf.placeholder(shape=[None], dtype=tf.string)

  features = tf.parse_example(
    receiver_tensors,
    features=get_feature_spec(is_serving=True)
  )

  for key in features:
    features[key] = tf.expand_dims(features[key], -1)

  return tf.estimator.export.ServingInputReceiver(
    features=process_features(features),
    receiver_tensors={'example_proto': receiver_tensors}
  )


SERVING_INPUT_RECEIVER_FUNCTIONS = {
  'JSON': json_serving_input_receiver_fn,
  'EXAMPLE': example_serving_input_receiver_fn,
  'CSV': csv_serving_input_receiver_fn
}


# ******************************************************************************
# EVALUATING INPUT FUNCTIONS - YOU NEED NOT TO CHANGE THE FOLLOWING PART
# ******************************************************************************


def csv_evaluating_input_receiver_fn():
  """Creating an EvalInputReceiver object for CSV data.

  Returns:
    EvalInputReceiver
  """

  # Notice that the inputs are raw features, not transformed features.
  csv_row = tf.placeholder(shape=[None], dtype=tf.string)
  features = parse_csv(csv_row, is_serving=False)
  target = features.pop(metadata.TARGET_NAME)

  return tfma.export.EvalInputReceiver(
    features=process_features(features),
    receiver_tensors={'examples': csv_row},
    labels=target)


def example_evaluating_input_receiver_fn():
  """Creating an EvalInputReceiver object for TFRecords data.

  Returns:
      EvalInputReceiver
  """

  tf_example = tf.placeholder(shape=[None], dtype=tf.string)
  features = tf.parse_example(
    tf_example,
    features=get_feature_spec(is_serving=False))

  for key in features:
    features[key] = tf.expand_dims(features[key], -1)

  return tfma.export.EvalInputReceiver(
    features=process_features(features),
    receiver_tensors={'examples': tf_example},
    labels=features[metadata.TARGET_NAME])


EVALUATING_INPUT_RECEIVER_FUNCTIONS = {
  'EXAMPLE': example_evaluating_input_receiver_fn,
  'CSV': csv_evaluating_input_receiver_fn
}

