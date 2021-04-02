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


import math
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as feature_column
import metadata


# ******************************************************************************
# YOU MAY CHANGE THIS FUNCTION TO ADD EXTENDED FEATURES
# ******************************************************************************


def _extend_feature_columns(feature_columns, args):
  """ Use to define additional feature columns, such as bucketized_column(s),
  crossed_column(s), and embedding_column(s). args can be used to parameterise
  the creation of the extended columns (e.g., number of buckets, etc.).

  Default behaviour is to return the original feature_columns list as-is.

  Args:
      feature_columns: list of feature_columns.
      args: experiment parameters.
  Returns:
      list of extended feature_columns
  """

  # # examples - given:
  # 'x' and 'y' are two numeric features:
  # 'alpha' and 'beta' are two categorical features
  #
  # # crossing
  # alpha_X_beta = tf.feature_column.crossed_column(
  #  [feature_columns['alpha'], feature_columns['beta']], 4)
  #
  # # bucketization
  # num_buckets = args.num_buckets
  # buckets = np.linspace(-2, 2, num_buckets).tolist()
  #
  # x_bucketized = tf.feature_column.bucketized_column(
  #  feature_columns['x'], buckets)
  #
  # y_bucketized = tf.feature_column.bucketized_column(
  #  feature_columns['y'], buckets)
  #
  # # crossing bucketized columns
  # x_bucketized_X_y_bucketized = tf.feature_column.crossed_column(
  #  [x_bucketized, y_bucketized], int(1e4))
  #
  # # embedding
  # x_bucketized_X_y_bucketized_embedded = tf.feature_column.embedding_column(
  #  x_bucketized_X_y_bucketized, dimension=args.embedding_size)

  extended_feature_columns = []

  for column_name in feature_columns:
    column = feature_columns[column_name]

    if isinstance(column, feature_column.VocabularyListCategoricalColumn):
      # Embed the categorical feature
      if args.embed_categorical_columns:
        vocab_size = len(column.vocabulary_list)
        extended_feature_columns.append(
          tf.feature_column.embedding_column(column,
            dimension=math.ceil(math.sqrt(vocab_size))))
      # Convert the categorical feature to indicator
      if args.use_indicator_columns:
        extended_feature_columns.append(
          tf.feature_column.indicator_column(column))

    if isinstance(column, feature_column.IdentityCategoricalColumn):
      # Embed the categorical feature
      if args.embed_categorical_columns:
        vocab_size = column.num_buckets
        extended_feature_columns.append(
          tf.feature_column.embedding_column(column,
            dimension=math.ceil(math.sqrt(vocab_size))))
      # Convert the categorical feature to indicator
      if args.use_indicator_columns:
        extended_feature_columns.append(
          tf.feature_column.indicator_column(column))

    if isinstance(column, feature_column.HashedCategoricalColumn):
      # Convert the categorical feature to indicator
      if args.use_indicator_columns:
        extended_feature_columns.append(
          tf.feature_column.indicator_column(column))

    # Add numeric features
    if isinstance(column, feature_column.NumericColumn):
      extended_feature_columns.append(column)
    # Only add the sparse feature as-is if args.use_wide_columns is set to True
    elif args.use_wide_columns:
      extended_feature_columns.append(column)

  return extended_feature_columns


# ******************************************************************************
# YOU NEED NOT TO CHANGE THESE FUNCTIONS TO CREATE THE WIDE AND DEEP FEATURES
# ******************************************************************************

def _create_feature_columns():
  """Create TensorFlow feature_column(s) based on the metadata.

  The TensorFlow feature_column objects are created based on the data types of
  the features defined in the metadata.py module.

  The feature_column(s) are created based on the input features,
  and the constructed features (process_features method in input.py),
  during reading data files. Both type of features (input and constructed)
  should be in metadata.

  Returns:
    dictionary of name:feature_column
  """

  feature_columns = {}

  # Add numeric features
  for feature_name in metadata.NUMERIC_FEATURE_NAMES_WITH_STATS:
    try:
      mean = metadata.NUMERIC_FEATURE_NAMES_WITH_STATS['mean']
      variance = metadata.NUMERIC_FEATURE_NAMES_WITH_STATS['var']
      normalizer_fn = lambda value: (value - mean) / math.sqrt(variance)
    except:
      normalizer_fn = None
    feature_columns[feature_name] = (
      tf.feature_column.numeric_column(
        feature_name, normalizer_fn=normalizer_fn))

  # Add categorical columns with identity
  for feature_name in metadata.CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY:
    feature_columns[feature_name] = (
      tf.feature_column.categorical_column_with_identity(
        feature_name,
        num_buckets=metadata.CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY[
          feature_name])
    )

  # Add categorical columns with vocabulary
  for feature_name in metadata.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY:
    feature_columns[feature_name] = (
      tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name,
        vocabulary_list=metadata.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY[
          feature_name])
    )

  # Add categorical columns with hash bucket
  for feature_name in metadata.CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET:
    feature_columns[feature_name] = (
      tf.feature_column.categorical_column_with_hash_bucket(
        feature_name,
        hash_bucket_size=metadata.CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET[
          feature_name])
    )

  return feature_columns


def _get_sparse_and_dense_columns(feature_columns):
  """Separates the spares from the dense feature columns.
  Args:
    feature_columns: list of feature columns
  Return:
    sparse_columns, dense_columns
  """

  dense_columns = [
    column for column in feature_columns
    if (
      isinstance(column, feature_column.NumericColumn) or
      isinstance(column, feature_column.EmbeddingColumn) or
      isinstance(column, feature_column.IndicatorColumn))
  ]

  sparse_columns = [
    column for column in feature_columns
    if(
      isinstance(column, feature_column.VocabularyListCategoricalColumn) or
      isinstance(column, feature_column.IdentityCategoricalColumn) or
      isinstance(column, feature_column.BucketizedColumn) or
      isinstance(column, feature_column.CrossedColumn))
  ]

  return sparse_columns, dense_columns


def create_wide_and_deep_columns(args):
  """Creates wide and deep feature_column lists.
  Args:
    args: experiment parameters.
  Returns
    wide_columns, deep_columns
  """

  # Create base feature columns
  feature_columns = _create_feature_columns()
  # Extend feature columns
  feature_columns = _extend_feature_columns(feature_columns, args)
  # Separate sparse from dense columns
  wide_columns, deep_columns = _get_sparse_and_dense_columns(feature_columns)
  return wide_columns, deep_columns
