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

import tensorflow as tf

# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
    # Categorical base columns

    # For categorical columns with known values we can provide lists
    # of values ahead of time.
    tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', [' Female', ' Male']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'race', [
            ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other',
            ' White'
        ]),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            ' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
            ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',
            ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th',
            ' Preschool', ' 12th'
        ]),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            ' Married-civ-spouse', ' Divorced', ' Married-spouse-absent',
            ' Never-married', ' Separated', ' Married-AF-spouse', ' Widowed'
        ]),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            ' Husband', ' Not-in-family', ' Wife', ' Own-child', ' Unmarried',
            ' Other-relative'
        ]),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            ' Self-emp-not-inc', ' Private', ' State-gov', ' Federal-gov',
            ' Local-gov', ' ?', ' Self-emp-inc', ' Without-pay', ' Never-worked'
        ]),

    # For columns with a large number of values, or unknown values
    # We can use a hash function to convert to categories.
    tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=100, dtype=tf.string),
    tf.feature_column.categorical_column_with_hash_bucket(
        'native_country', hash_bucket_size=100, dtype=tf.string),

    # Continuous base columns.
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('education_num'),
    tf.feature_column.numeric_column('capital_gain'),
    tf.feature_column.numeric_column('capital_loss'),
    tf.feature_column.numeric_column('hours_per_week'),
]


def get_deep_and_wide_columns(embedding_size=8):
    """Creates deep and wide feature_column lists.
    Args:
        embedding_size: (int), the number of dimensions used to represent
        categorical features when providing them as inputs to the DNN.
    Returns:
        [tf.feature_column],[tf.feature_column]: deep and wide feature_column
            lists.
    """

    (gender, race, education, marital_status, relationship, workclass,
     occupation,
     native_country, age, education_num, capital_gain, capital_loss,
     hours_per_week) = INPUT_COLUMNS

    # Reused Transformations.
    # Continuous columns can be converted to categorical via bucketization
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    wide_columns = [
        # Interactions between different categorical features can also
        # be added as new virtual features.
        tf.feature_column.crossed_column(['education', 'occupation'],
                                         hash_bucket_size=int(1e4)),
        tf.feature_column.crossed_column([age_buckets, race, 'occupation'],
                                         hash_bucket_size=int(1e6)),
        tf.feature_column.crossed_column(['native_country', 'occupation'],
                                         hash_bucket_size=int(1e4)),
        gender,
        native_country,
        education,
        occupation,
        workclass,
        marital_status,
        relationship,
        age_buckets,
    ]

    deep_columns = [
        # Use indicator columns for low dimensional vocabularies
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(gender),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.indicator_column(race),

        # Use embedding columns for high dimensional vocabularies
        tf.feature_column.embedding_column(
            native_country, dimension=embedding_size),
        tf.feature_column.embedding_column(
            occupation, dimension=embedding_size),
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
    ]

    return deep_columns, wide_columns
