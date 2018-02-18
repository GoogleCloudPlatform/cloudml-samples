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

# ************************************************************************
# YOU NEED TO MODIFY THE META DATA TO ADAPT THE TRAINER TEMPLATE YOUR DATA
# ************************************************************************

# task type can be either 'classification' or 'regression', based on the target feature in the dataset
TASK_TYPE = 'classification'

# list of all the columns (header) of the input data file(s)
HEADER = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
          'marital_status', 'occupation', 'relationship', 'race', 'gender',
          'capital_gain', 'capital_loss', 'hours_per_week',
          'native_country', 'income_bracket']

# list of the default values of all the columns of the input data, to help decoding the data types of the columns
HEADER_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                   [0], [0], [0], [''], ['']]

# list of the feature names of type int or float
INPUT_NUMERIC_FEATURE_NAMES = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

# numeric features constructed, if any, in process_features function in input.py module,
# as part of reading data
CONSTRUCTED_NUMERIC_FEATURE_NAMES = []

# a dictionary of feature names with int values, but to be treated as categorical features.
# In the dictionary, the key is the feature name, and the value is the num_buckets (count of distinct values)
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# categorical features with identity constructed, if any, in process_features function in input.py module,
# as part of reading data. Usually include constructed boolean flags
CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# a dictionary of categorical features with few nominal values (to be encoded as one-hot indicators)
#  In the dictionary, the key is the feature name, and the value is the list of feature vocabulary
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {
    'gender': ['Female', 'Male'],

    'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'],

    'education': ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                  'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                  '5th-6th', '10th', '1st-4th', 'Preschool', '12th'],

    'marital_status': ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                       'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'],

    'relationship': ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],

    'workclass': ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov', 'Local-gov', '?',
                  'Self-emp-inc', 'Without-pay', 'Never-worked']
}

# a dictionary of categorical features with many values (sparse features)
# In the dictionary, the key is the feature name, and the value is the bucket size
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {
    'occupation': 50,
    'native_country': 100
}

# list of all the categorical feature names
INPUT_CATEGORICAL_FEATURE_NAMES = list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

# list of all the input feature names to be used in the model
INPUT_FEATURE_NAMES = INPUT_NUMERIC_FEATURE_NAMES + INPUT_CATEGORICAL_FEATURE_NAMES

# the column include the weight of each record
WEIGHT_COLUMN_NAME = 'fnlwgt'

# target feature name (response or class variable)
TARGET_NAME = 'income_bracket'

# list of the class values (labels) in a classification dataset
TARGET_LABELS = ['<=50K', '>50K']

# list of the columns expected during serving (which probably different than the header of the training data)
SERVING_COLUMNS = ['age', 'workclass', 'education', 'education_num',
                   'marital_status', 'occupation', 'relationship', 'race', 'gender',
                   'capital_gain', 'capital_loss', 'hours_per_week',
                   'native_country']

# list of the default values of all the columns of the serving data, to help decoding the data types of the columns
SERVING_DEFAULTS = [[0], [''], [''], [0], [''], [''], [''], [''], [''], [0], [0], [0], ['']]
