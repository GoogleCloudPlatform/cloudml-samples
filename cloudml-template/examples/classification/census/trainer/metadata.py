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
COLUMN_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
          'marital_status', 'occupation', 'relationship', 'race', 'gender',
          'capital_gain', 'capital_loss', 'hours_per_week',
          'native_country', 'income_bracket']

# List of the columns expected during serving (which is probably different to
# the header of the training data).
SERVING_COLUMN_NAMES = [
  'age', 'workclass', 'education', 'education_num',
  'marital_status', 'occupation', 'relationship', 'race', 'gender',
  'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']

# List of the default values of all the columns present in the input data.
# This helps decoding the data types of the columns.
DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                   [0], [0], [0], [''], ['']]

# Dictionary of the feature names of type int or float. In the dictionary,
# the key is the feature name, and the value is another dictionary includes
# the mean and the variance of the numeric features.
# E.g. {feature_1: {mean: 0, variance:1}, feature_2: {mean: 10, variance:3}}
# The value can be set to None if you don't want to not normalize.
NUMERIC_FEATURE_NAMES_WITH_STATS = {
  'age': None, 'education_num': None, 'capital_gain': None,
  'capital_loss': None, 'hours_per_week': None
}

# Dictionary of feature names with int values, but to be treated as
# categorical features. In the dictionary, the key is the feature name,
# and the value is the num_buckets (count of distinct values).
CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}


# Dictionary of categorical features with few nominal values. In the dictionary,
# the key is the feature name, and the value is the list of feature vocabulary.
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {
  'gender': ['Female', 'Male'],
  'race': [
    'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'
  ],
  'education': [
    'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
    'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
    '5th-6th', '10th', '1st-4th', 'Preschool', '12th'
  ],
  'marital_status': [
    'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Never-married',
    'Separated', 'Married-AF-spouse', 'Widowed'
  ],
  'relationship': [
    'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
    'Other-relative'
  ],
  'workclass': [
    'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov', 'Local-gov', '?',
                  'Self-emp-inc', 'Without-pay', 'Never-worked'
  ]
}

# Dictionary of categorical features with many values. In the dictionary,
# the key is the feature name, and the value is the number of buckets.
CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {
    'occupation': 50,
    'native_country': 100
}

# Column includes the relative weight of each record.
WEIGHT_COLUMN_NAME = 'fnlwgt'

# Target feature name (response or class variable).
TARGET_NAME = 'income_bracket'

# List of the class values (labels) in a classification dataset.
TARGET_LABELS = ['<=50K', '>50K']