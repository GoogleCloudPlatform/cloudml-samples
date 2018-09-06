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


import tensorflow as tf
from tensorflow.python.feature_column import feature_column

import metadata
import task
import input


# **************************************************************************
# YOU MAY IMPLEMENT THIS FUNCTION TO ADD EXTENDED FEATURES
# **************************************************************************


def extend_feature_columns(feature_columns):
    """ Use to define additional feature columns, such as bucketized_column(s), crossed_column(s),
    and embedding_column(s). task.HYPER_PARAMS can be used to parameterise the creation
    of the extended columns (e.g., embedding dimensions, number of buckets, etc.).

    Default behaviour is to return the original feature_columns list as-is.

    Args:
        feature_columns: {column_name: tf.feature_column} - dictionary of base feature_column(s)
    Returns:
        {string: tf.feature_column}: extended feature_column(s) dictionary
    """

    # # examples - given:
    # 'x' and 'y' are two numeric features:
    # 'alpha' and 'beta' are two categorical features
    #
    # # crossing
    # feature_columns['alpha_X_beta'] = tf.feature_column.crossed_column(
    #     [feature_columns['alpha'], feature_columns['beta']], 4)
    #
    # # bucketization
    # num_buckets = task.HYPER_PARAMS.num_buckets
    # buckets = np.linspace(-2, 2, num_buckets).tolist()
    #
    # feature_columns['x_bucketized'] = tf.feature_column.bucketized_column(
    #     feature_columns['x'], buckets)
    #
    # feature_columns['y_bucketized'] = tf.feature_column.bucketized_column(
    #     feature_columns['y'], buckets)
    #
    # # crossing bucketized columns
    # feature_columns['x_bucketized_X_y_bucketized'] = tf.feature_column.crossed_column(
    #     [feature_columns['x_bucketized'], feature_columns['y_bucketized']], int(1e4))
    #
    # # embedding
    # feature_columns['x_bucketized_X_y_bucketized_embedded'] = tf.feature_column.embedding_column(
    #     feature_columns['x_bucketized_X_y_bucketized'], dimension=task.HYPER_PARAMS.embedding_size)

    return feature_columns


# **************************************************************************
# YOU MAY NOT CHANGE THIS FUNCTION TO CREATE FEATURE COLUMNS
# **************************************************************************


def create_feature_columns():
    """Creates tensorFlow feature_column(s) based on the metadata of the input features.

    The tensorFlow feature_column objects are created based on the data types of the features
    defined in the metadata.py module.

    The feature_column(s) are created based on the input features,
    and the constructed features (process_features method in input.py), during reading data files.
    Both type of features (input and constructed) should be defined in metadata.py.

    Extended features (if any) are created, based on the base features, as the extend_feature_columns
    method is called, before the returning complete the feature_column dictionary.

    Returns:
      {string: tf.feature_column}: dictionary of name:feature_column .
    """

    # load the numeric feature stats (if exists)
    feature_stats = input.load_feature_stats()

    # all the numerical features including the input and constructed ones
    numeric_feature_names = set(metadata.INPUT_NUMERIC_FEATURE_NAMES + metadata.CONSTRUCTED_NUMERIC_FEATURE_NAMES)

    # create t.feature_column.numeric_column columns without scaling
    if feature_stats is None:
        numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name, normalizer_fn=None)
                           for feature_name in numeric_feature_names}

    # create t.feature_column.numeric_column columns with scaling
    else:
        numeric_columns = {}

        for feature_name in numeric_feature_names:
            try:
                # standard scaling
                mean = feature_stats[feature_name]['mean']
                stdv = feature_stats[feature_name]['stdv']
                normalizer_fn = lambda x: (x - mean) / stdv

                # max_min scaling
                # min_value = feature_stats[feature_name]['min']
                # max_value = feature_stats[feature_name]['max']
                # normalizer_fn = lambda x: (x-min_value)/(max_value-min_value)

                numeric_columns[feature_name] = tf.feature_column.numeric_column(feature_name,
                                                                                 normalizer_fn=normalizer_fn)
            except:
                numeric_columns[feature_name] = tf.feature_column.numeric_column(feature_name,
                                                                                 normalizer_fn=None)

    # all the categorical features with identity including the input and constructed ones
    categorical_feature_names_with_identity = metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY
    categorical_feature_names_with_identity.update(metadata.CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY)

    # create tf.feature_column.categorical_column_with_identity columns
    categorical_columns_with_identity = \
        {item[0]: tf.feature_column.categorical_column_with_identity(item[0], item[1])
         for item in categorical_feature_names_with_identity.items()}

    # create tf.feature_column.categorical_column_with_vocabulary_list columns
    categorical_columns_with_vocabulary = \
        {item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
         for item in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}

    # create tf.feature_column.categorical_column_with_hash_bucket columns
    categorical_columns_with_hash_bucket = \
        {item[0]: tf.feature_column.categorical_column_with_hash_bucket(item[0], item[1])
         for item in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.items()}

    # this will include all the feature columns of various types
    feature_columns = {}

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_columns_with_identity is not None:
        feature_columns.update(categorical_columns_with_identity)

    if categorical_columns_with_vocabulary is not None:
        feature_columns.update(categorical_columns_with_vocabulary)

    if categorical_columns_with_hash_bucket is not None:
        feature_columns.update(categorical_columns_with_hash_bucket)

    # add extended feature_column(s) before returning the complete feature_column dictionary
    return extend_feature_columns(feature_columns)


# **************************************************************************
# YOU NEED NOT TO CHANGE THIS FUNCTION TO DEFINE WIDE AND DEEP COLUMNS
# **************************************************************************


def get_deep_and_wide_columns(feature_columns, use_indicators=True, use_wide_columns=True):
    """Creates deep and wide feature_column lists.

    Given a list of feature_column(s), each feature_column is categorised as either:
    1) dense, if the column is tf.feature_column._NumericColumn or feature_column._EmbeddingColumn,
    2) categorical, if the column is tf.feature_column._VocabularyListCategoricalColumn or
    tf.feature_column._BucketizedColumn, or
    3) sparse, if the column is tf.feature_column._HashedCategoricalColumn or tf.feature_column._CrossedColumn.

    If use_indicators=True, then categorical_columns are converted into indicator_columns, and used as dense features
    in the deep part of the model. if use_wide_columns=True, then categorical_columns are used as sparse features
    in the wide part of the model.

    deep_columns = dense_columns + indicator_columns
    wide_columns = categorical_columns + sparse_columns

    Args:
        feature_columns: [tf.feature_column] - A list of tf.feature_column objects.
        use_indicators: bool - if True, categorical_columns are converted into tf.feature_column.indicator_column(s).
        use_wide_columns: bool - if True, categorical_columns are treated wide columns.
    Returns:
        [tf.feature_column],[tf.feature_column]: deep and wide feature_column lists.
    """
    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn) |
                              isinstance(column, feature_column._EmbeddingColumn),
               feature_columns)
    )

    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._IdentityCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn),
               feature_columns)
    )

    sparse_columns = list(
        filter(lambda column: isinstance(column, feature_column._HashedCategoricalColumn) |
                              isinstance(column, feature_column._CrossedColumn),
               feature_columns)
    )

    indicator_columns = []

    if use_indicators:
        indicator_columns = list(
            map(lambda column: tf.feature_column.indicator_column(column),
                categorical_columns)
        )

    deep_columns = dense_columns + indicator_columns
    wide_columns = sparse_columns + (categorical_columns if use_wide_columns else [])

    return deep_columns, wide_columns
