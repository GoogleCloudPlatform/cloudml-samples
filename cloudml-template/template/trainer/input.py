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


import json
import multiprocessing

import tensorflow as tf
from tensorflow import data
from tensorflow_model_analysis import export as tfma_export

import metadata
import featurizer
import task


# **************************************************************************
# YOU NEED NOT TO CHANGE THESE FUNCTIONS TO PARSE THE INPUT RECORDS
# **************************************************************************


def parse_csv(csv_row, is_serving=False):
    """Takes a batch of csv rows and returns a dict of rank-2 tensors.

    Takes a rank-2 tensor and converts it into rank-2 tensor, with respect to
    its data type (inferred from the metadata)

    Args:
        csv_row: rank-2 tensor of type string (csv rows)
        is_serving: boolean to indicate whether this function is called during
        serving or training since the serving csv_row input is different than
        the training input (i.e., no target column)
    Returns:
        rank-2 tensors' dictionary of the correct data type
    """

    if is_serving:
        column_names = metadata.SERVING_COLUMNS
        defaults = metadata.SERVING_DEFAULTS
    else:
        column_names = metadata.HEADER
        defaults = metadata.HEADER_DEFAULTS

    defaults = [[default] for default in defaults]
    columns = tf.decode_csv(csv_row, record_defaults=defaults)
    features = dict(zip(column_names, columns))

    return features


def parse_tf_example(example_proto, is_serving=False):
    """Takes a batch of string input tensors (example proto) and returns a dict of rank-2 tensors.

    Takes a rank-2 tensor and converts it into a dictionary of rank-2 tensor, with respect to its data type
    (inferred from the  metadata)

    Args:
        example_proto: rank-2 tensor of type string (example proto)
        is_serving: boolean to indicate whether this function is called during serving or training
        since the serving csv_row input is different than the training input (i.e., no target column)
    Returns:
        rank-2 tensors' dictionary of the correct data type
    """

    feature_spec = {}

    for feature_name in metadata.INPUT_FEATURE_NAMES:
        feature_spec[feature_name] = tf.FixedLenFeature(shape=1, dtype=tf.float32)

    for feature_name in metadata.INPUT_CATEGORICAL_FEATURE_NAMES:
        feature_spec[feature_name] = tf.FixedLenFeature(shape=1, dtype=tf.string)

    if not is_serving:

        if metadata.TASK_TYPE == 'regression':
            feature_spec[metadata.TARGET_NAME] = tf.FixedLenFeature(shape=1, dtype=tf.float32)
        else:
            feature_spec[metadata.TARGET_NAME] = tf.FixedLenFeature(shape=(), dtype=tf.string)

    parsed_features = tf.parse_example(serialized=example_proto, features=feature_spec)

    return parsed_features


# **************************************************************************
# YOU MAY IMPLEMENT THIS FUNCTION FOR CUSTOM FEATURE ENGINEERING
# **************************************************************************


def process_features(features):
    """ Use to implement custom feature engineering logic, e.g. polynomial expansion, etc.

    Default behaviour is to return the original feature tensors dictionary as-is.

    Args:
        features: {string:tensors} - dictionary of feature tensors
    Returns:
        {string:tensors}: extended feature tensors dictionary
    """

    # examples - given:
    # 'x' and 'y' are two numeric features:
    # 'alpha' and 'beta' are two categorical features

    # create new features using custom logic
    # features['x_2'] = tf.pow(features['x'],2)
    # features['y_2'] = tf.pow(features['y'], 2)
    # features['xy'] = features['x'] * features['y']
    # features['sin_x'] = tf.sin(features['x'])
    # features['cos_y'] = tf.cos(features['x'])
    # features['log_xy'] = tf.log(features['xy'])
    # features['sqrt_xy'] = tf.sqrt(features['xy'])

    # create boolean flags
    # features['x_grt_y'] = tf.cast(features['x'] > features['y'], tf.int32)
    # features['alpha_eq_beta'] = features['alpha'] == features['beta']

    # add created features to metadata (if not already defined in metadata.py)
    # CONSTRUCTED_NUMERIC_FEATURE_NAMES += ['x_2', 'y_2', 'xy', ....]
    # CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY['x_grt_y'] = 2
    # CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY['alpha_eq_beta'] = 2

    return features


def get_features_target_tuple(features):
    """ Get a tuple of input feature tensors and target feature tensor.

    Args:
        features: {string:tensors} - dictionary of feature tensors
    Returns:
          {string:tensors}, {tensor} -  input feature tensor dictionary and target feature tensor
    """

    unused_features = list(set(metadata.HEADER) -
                           set(metadata.INPUT_FEATURE_NAMES) -
                           {metadata.TARGET_NAME} -
                           {metadata.WEIGHT_COLUMN_NAME})

    # remove unused columns (if any)
    for column in unused_features:
        features.pop(column, None)

    # get target feature
    target = features.pop(metadata.TARGET_NAME)

    return features, target


# **************************************************************************
# YOU NEED NOT TO CHANGE THIS FUNCTION TO READ DATA FILES
# **************************************************************************


def generate_input_fn(file_names_pattern,
                     file_encoding='csv',
                     mode=tf.estimator.ModeKeys.EVAL,
                     skip_header_lines=0,
                     num_epochs=1,
                     batch_size=200,
                     multi_threading=True):
    """Generates an input function for reading training and evaluation data file(s).
    This uses the tf.data APIs.

    Args:
        file_names_pattern: [str] - file name or file name patterns from which to read the data.
        mode: tf.estimator.ModeKeys - either TRAIN or EVAL.
            Used to determine whether or not to randomize the order of data.
        file_encoding: type of the text files. Can be 'csv' or 'tfrecords'
        skip_header_lines: int set to non-zero in order to skip header lines in CSV files.
        num_epochs: int - how many times through to read the data.
          If None will loop through data indefinitely
        batch_size: int - first dimension size of the Tensors returned by input_fn
        multi_threading: boolean - indicator to use multi-threading or not
    Returns:
        A function () -> (features, indices) where features is a dictionary of
          Tensors, and indices is a single Tensor of label indices.
    """
    def _input_fn():

        shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

        data_size = task.HYPER_PARAMS.train_size if mode == tf.estimator.ModeKeys.TRAIN else None

        num_threads = multiprocessing.cpu_count() if multi_threading else 1

        buffer_size = 2 * batch_size + 1

        print("")
        print("* data input_fn:")
        print("================")
        print("Mode: {}".format(mode))
        print("Input file(s): {}".format(file_names_pattern))
        print("Files encoding: {}".format(file_encoding))
        print("Data size: {}".format(data_size))
        print("Batch size: {}".format(batch_size))
        print("Epoch count: {}".format(num_epochs))
        print("Thread count: {}".format(num_threads))
        print("Shuffle: {}".format(shuffle))
        print("================")
        print("")

        file_names = tf.matching_files(file_names_pattern)

        if file_encoding == 'csv':
            # Parallely processes num_threads files at the time.
            # Also, since the batch function is called before the map function
            # we use a block lenght of 1 to output one batch from every input
            # file before moving to the next file
            dataset = (data.Dataset.from_tensor_slices(file_names)
              .interleave(lambda x:
                data.TextLineDataset(x)
                .skip(skip_header_lines)
                .batch(batch_size)
                .map(parse_csv, num_parallel_calls=num_threads),
                  cycle_length=num_threads, block_length=1))

        else:
            dataset = data.TFRecordDataset(filenames=file_names)
            dataset = dataset = dataset.batch(batch_size)
            dataset = dataset.map(lambda tf_examples: parse_tf_example(tf_examples),
                                  num_parallel_calls=num_threads)

        dataset = dataset.map(lambda features: get_features_target_tuple(features),
                              num_parallel_calls=num_threads)
        dataset = dataset.map(lambda features, target: (process_features(features), target),
                              num_parallel_calls=num_threads)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.prefetch(buffer_size)
        dataset = dataset.repeat(num_epochs)

        iterator = dataset.make_one_shot_iterator()
        features, target = iterator.get_next()

        return features, target

    return _input_fn


# **************************************************************************
# YOU MAY CHANGE THIS FUNCTION TO LOAD YOUR NUMERIC COLUMN STATS
# **************************************************************************


def load_feature_stats():
    """
    Load numeric column pre-computed statistics (mean, stdv, min, max, etc.)
    in order to be used for scaling/stretching numeric columns.

    In practice, the statistics of large datasets are computed prior to model training,
    using dataflow (beam), dataproc (spark), BigQuery, etc.

    The stats are then saved to gcs location. The location is passed to package
    in the --feature-stats-file argument. However, it can be a local path as well.

    Returns:
        json object with the following schema: stats['feature_name']['state_name']
    """

    feature_stats = None
    try:
        if task.HYPER_PARAMS.feature_stats_file is not None and tf.gfile.Exists(task.HYPER_PARAMS.feature_stats_file):
            with tf.gfile.Open(task.HYPER_PARAMS.feature_stats_file) as file:
                content = file.read()
            feature_stats = json.loads(content)
            print("INFO:feature stats were successfully loaded from local file...")
        else:
            print("WARN: feature stats file not found. numerical columns will not be normalised...")
    except:
        print("WARN: couldn't load feature stats. numerical columns will not be normalised...")

    return feature_stats


# ****************************************************************************
# SERVING FUNCTIONS - YOU NEED NOT TO CHANGE THE FOLLOWING PART
# ****************************************************************************


def json_serving_input_fn():
    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.INPUT_FEATURE_NAMES]

    inputs = {}

    for column in input_feature_columns:
        if column.name in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY:
            inputs[column.name] = tf.placeholder(shape=[None], dtype=tf.int32)
        else:
            inputs[column.name] = tf.placeholder(shape=[None], dtype=column.dtype)

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }

    return tf.estimator.export.ServingInputReceiver(
        features=process_features(features),
        receiver_tensors=inputs
    )


def csv_serving_input_fn():
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )

    features = parse_csv(csv_row, is_serving=True)

    unused_features = list(set(metadata.SERVING_COLUMNS) - set(metadata.INPUT_FEATURE_NAMES) - {metadata.TARGET_NAME})

    # Remove unused columns (if any)
    for column in unused_features:
        features.pop(column, None)

    return tf.estimator.export.ServingInputReceiver(
        features=process_features(features),
        receiver_tensors={'csv_row': csv_row}
    )


def tfma_csv_serving_input_fn():
    """Build everything needed to run tf-model-analysis when using CSV as input
    for the model.

    Returns:
        EvalInputReceiver function, which contains:
          - Tensorflow graph which parses raw untranformed features, applies the
            tf-transform preprocessing operators.
          - Set of raw, untransformed features.
          - Label against which predictions will be compared.
    """

    # Notice that the inputs are raw features, not transformed features here.
    csv_row = tf.placeholder(shape=[None], dtype=tf.string)

    features = parse_csv(csv_row, is_serving=False)

    # The key name MUST be 'examples'. See https://goo.gl/2SV7Ug
    receiver_tensors = {'examples': csv_row}

    return tfma_export.EvalInputReceiver(
        features=process_features(features),
        receiver_tensors=receiver_tensors,
        labels=features[metadata.TARGET_NAME])


def example_serving_input_fn():
    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.INPUT_FEATURE_NAMES]

    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    feature_scalars = tf.parse_example(
        example_bytestring,
        tf.feature_column.make_parse_example_spec(input_feature_columns)
    )

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_scalars.iteritems()
    }

    return tf.estimator.export.ServingInputReceiver(
        features=process_features(features),
        receiver_tensors={'example_proto': example_bytestring}
    )


def tfma_example_serving_input_fn():
    """Build everything needed to run tf-model-analysis when using examples as
    input for the model.

    Returns:
        EvalInputReceiver function, which contains:
          - Tensorflow graph which parses raw untranformed features, applies the
            tf-transform preprocessing operators.
          - Set of raw, untransformed features.
          - Label against which predictions will be compared.
    """
    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.INPUT_FEATURE_NAMES]

    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    feature_scalars = tf.parse_example(
        example_bytestring,
        tf.feature_column.make_parse_example_spec(input_feature_columns)
    )

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_scalars.iteritems()
    }

    return tfma_export.EvalInputReceiver(
        features=process_features(features),
        # The key name MUST be 'examples'. See https://goo.gl/2SV7Ug
        receiver_tensors={'examples': example_bytestring},
        labels=features[metadata.TARGET_NAME])


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}

TFMA_SERVING_FUNCTIONS = {
    'EXAMPLE': tfma_example_serving_input_fn,
    'CSV': tfma_csv_serving_input_fn
}

