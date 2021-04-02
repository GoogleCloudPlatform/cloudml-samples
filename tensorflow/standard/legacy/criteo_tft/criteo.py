# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Schema and tranform definition for the Criteo dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import coders
from tensorflow_transform.tf_metadata import dataset_schema


INTEGER_COLUMN_NAMES = [
    'int-feature-{}'.format(column_idx) for column_idx in range(1, 14)]
CATEGORICAL_COLUMN_NAMES = [
    'categorical-feature-{}'.format(column_idx) for column_idx in range(14, 40)]


def make_csv_coder(schema, mode=tf.contrib.learn.ModeKeys.TRAIN,
                   delimiter='\t'):
  """Produces a CsvCoder (with tab as the delimiter) from a data schema.

  Args:
    schema: A tf.Transform `Schema` object.
    mode: tf.contrib.learn.ModeKeys specifying if the source is being used for
      train/eval or prediction.
    delimiter: The delimiter used to construct the CsvCoder.

  Returns:
    A tf.Transform CsvCoder.
  """
  column_names = [] if mode == tf.contrib.learn.ModeKeys.INFER else ['clicked']
  for name in INTEGER_COLUMN_NAMES:
    column_names.append(name)
  for name in CATEGORICAL_COLUMN_NAMES:
    column_names.append(name)

  return coders.CsvCoder(column_names, schema, delimiter=delimiter)


def make_input_schema(mode=tf.contrib.learn.ModeKeys.TRAIN):
  """Input schema definition.

  Args:
    mode: tf.contrib.learn.ModeKeys specifying if the schema is being used for
      train/eval or prediction.
  Returns:
    A `Schema` object.
  """
  result = ({} if mode == tf.contrib.learn.ModeKeys.INFER
            else {'clicked': tf.FixedLenFeature(shape=[], dtype=tf.int64)})
  for name in INTEGER_COLUMN_NAMES:
    result[name] = tf.FixedLenFeature(
        shape=[], dtype=tf.int64, default_value=-1)
  for name in CATEGORICAL_COLUMN_NAMES:
    result[name] = tf.FixedLenFeature(shape=[], dtype=tf.string,
                                      default_value='')

  return dataset_schema.from_feature_spec(result)


def make_preprocessing_fn(frequency_threshold):
  """Creates a preprocessing function for criteo.

  Args:
    frequency_threshold: The frequency_threshold used when generating
      vocabularies for the categorical features.

  Returns:
    A preprocessing function.
  """
  def preprocessing_fn(inputs):
    """User defined preprocessing function for criteo columns.

    Args:
      inputs: dictionary of input `tensorflow_transform.Column`.
    Returns:
      A dictionary of `tensorflow_transform.Column` representing the transformed
          columns.
    """
    # TODO(b/35001605) Make this "passthrough" more DRY.
    result = {'clicked': inputs['clicked']}
    for name in INTEGER_COLUMN_NAMES:
      result[name] = inputs[name]
    for name in CATEGORICAL_COLUMN_NAMES:
      result[name + '_id'] = tft.string_to_int(
          inputs[name], frequency_threshold=frequency_threshold)

    return result

  return preprocessing_fn
