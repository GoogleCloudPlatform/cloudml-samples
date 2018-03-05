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

# ************************************************************************************
# YOU NEED TO MODIFY THE FOLLOWING METADATA TO ADAPT THE TRAINER TEMPLATE TO YOUR DATA
# ************************************************************************************

# Task type can be either 'classification', 'regression', or 'custom'
# This is based on the target feature in the dataset, and whether you use a canned or a custom estimator
TASK_TYPE = ''  # classification | regression | custom

# A List of all the columns (header) present in the input data file(s) in order to parse it.
# Note that, not all the columns present here will be input features to your model.
HEADER = []

# List of the default values of all the columns present in the input data.
# This helps decoding the data types of the columns.
HEADER_DEFAULTS = []

# List of the feature names of type int or float.
INPUT_NUMERIC_FEATURE_NAMES = []

# Numeric features constructed, if any, in process_features function in input.py module,
# as part of reading data.
CONSTRUCTED_NUMERIC_FEATURE_NAMES = []

# Dictionary of feature names with int values, but to be treated as categorical features.
# In the dictionary, the key is the feature name, and the value is the num_buckets (count of distinct values).
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# Categorical features with identity constructed, if any, in process_features function in input.py module,
# as part of reading data. Usually include constructed boolean flags.
CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# Dictionary of categorical features with few nominal values (to be encoded as one-hot indicators).
# In the dictionary, the key is the feature name, and the value is the list of feature vocabulary.
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {}

# Dictionary of categorical features with many values (sparse features).
# In the dictionary, the key is the feature name, and the value is the bucket size.
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

# List of all the categorical feature names.
# This is programmatically created based on the previous inputs.
INPUT_CATEGORICAL_FEATURE_NAMES = list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

# List of all the input feature names to be used in the model.
# This is programmatically created based on the previous inputs.
INPUT_FEATURE_NAMES = INPUT_NUMERIC_FEATURE_NAMES + INPUT_CATEGORICAL_FEATURE_NAMES

# Column includes the relative weight of each record.
WEIGHT_COLUMN_NAME = None

# Target feature name (response or class variable).
TARGET_NAME = ''

# List of the class values (labels) in a classification dataset.
TARGET_LABELS = []

# List of the columns expected during serving (which is probably different to the header of the training data).
SERVING_COLUMNS = []

# List of the default values of all the columns of the serving data.
# This helps decoding the data types of the columns.
SERVING_DEFAULTS = []
