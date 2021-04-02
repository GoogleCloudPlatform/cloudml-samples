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

# ******************************************************************************
# YOU NEED TO MODIFY THE FOLLOWING METADATA TO ADAPT THE TEMPLATE TO YOUR DATA
# ******************************************************************************

# Task type can be either 'classification', 'regression', or 'custom'.
# This is based on the target feature in the dataset.
TASK_TYPE = 'classification'

# List of all the columns (header) present in the input data file(s).
# Used for parsing the input data.
COLUMN_NAMES = [
  'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 
  'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'class']

# List of the columns expected during serving (which is probably different to
# the header of the training data).
SERVING_COLUMN_NAMES = [
  'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 
  'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20']

# List of the default values of all the columns present in the input data.
# This helps decoding the data types of the columns.
DEFAULTS = [
  ['?'], [0.0], ['?'], ['?'], [0.0], ['?'], ['?'], [0.0], ['?'], ['?'], [0.0], 
  ['?'], [0.0], ['?'], ['?'], [0.0], ['?'], [0.0], ['?'], ['?'], ['?']]

# Dictionary of the feature names of type int or float. In the dictionary,
# the key is the feature name, and the value is another dictionary includes
# the mean and the variance of the numeric features.
# E.g. {feature_1: {mean: 0, variance:1}, feature_2: {mean: 10, variance:3}}
# The value can be set to None if you don't want to not normalize.
NUMERIC_FEATURE_NAMES_WITH_STATS = {
  'A2': None, 'A5': None, 'A8': None, 'A11': None, 
  'A13': None, 'A16': None, 'A18':None}

# Dictionary of feature names with int values, but to be treated as
# categorical features. In the dictionary, the key is the feature name,
# and the value is the num_buckets (count of distinct values).
CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# Dictionary of categorical features with few nominal values. In the dictionary,
# the key is the feature name, and the value is the list of feature vocabulary.
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {
     'A1': ['A11', 'A12', 'A13', 'A14'],
     'A3': ['A30', 'A31', 'A32', 'A33', 'A34'],
     'A4': ['A40', 'A41', 'A42', 'A43', 'A44', 
            'A45', 'A46', 'A47', 'A48', 'A49', 'A410'],
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

# Dictionary of categorical features with many values. In the dictionary,
# the key is the feature name, and the value is the number of buckets.
CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

# Column includes the relative weight of each record.
WEIGHT_COLUMN_NAME = None

# Target feature name (response or class variable).
TARGET_NAME = 'class'

# List of the class values (labels) in a classification dataset.
TARGET_LABELS = ['Good', 'Bad']

