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
TASK_TYPE = 'custom'

# list of all the columns (header) of the input data file(s)
HEADER = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
          'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
          'class']

# list of the default values of all the columns of the input data, to help decoding the data types of the columns
HEADER_DEFAULTS = [['?'], [0.0], ['?'], ['?'], [0.0], ['?'], ['?'], [0.0], ['?'], ['?'],
                   [0.0], ['?'], [0.0], ['?'], ['?'], [0.0], ['?'], [0.0], ['?'], ['?'],
                   ['?']]

# list of the feature names of type int or float
INPUT_NUMERIC_FEATURE_NAMES = ['A2', 'A5', 'A8', 'A11', 'A13', 'A16', 'A18']

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
     'A1': ['A11', 'A12', 'A13', 'A14'],
     'A3': ['A30', 'A31', 'A32', 'A33', 'A34'],
     'A4': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'],
     'A6': ['A61', 'A62', 'A63', 'A64', 'A65'],
     'A7': ['A71', 'A72', 'A73', 'A74', 'A75'],
     'A9': ['A91', 'A92', 'A93', 'A94', 'A95'],
    'A10': ['A101', 'A102', 'A103'],
    'A12': ['A121', 'A122', 'A123', 'A124'],
    'A14': ['A141', 'A142', 'A143'],
    'A15': ['A151', 'A152', 'A153'],
    'A17': ['A171', 'A172', 'A173', 'A174'],
    'A19': ['A191', 'A192'],
    'A20': ['A201', 'A202']
}

# a dictionary of categorical features with many values (sparse features)
# In the dictionary, the key is the feature name, and the value is the bucket size
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

# list of all the categorical feature names
INPUT_CATEGORICAL_FEATURE_NAMES = list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

# list of all the input feature names to be used in the model
INPUT_FEATURE_NAMES = INPUT_NUMERIC_FEATURE_NAMES + INPUT_CATEGORICAL_FEATURE_NAMES

# the column include the weight of each record
WEIGHT_COLUMN_NAME = None

# target feature name (response or class variable)
TARGET_NAME = 'class'

# list of the class values (labels) in a classification dataset
TARGET_LABELS = ['Good', 'Bad']

# list of the columns expected during serving (which probably different than the header of the training data)
SERVING_COLUMNS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                   'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20']

# list of the default values of all the columns of the serving data, to help decoding the data types of the columns
SERVING_DEFAULTS = [['?'], [0.0], ['?'], ['?'], [0.0], ['?'], ['?'], [0.0], ['?'], ['?'],
                   [0.0], ['?'], [0.0], ['?'], ['?'], [0.0], ['?'], [0.0], ['?'], ['?']]
