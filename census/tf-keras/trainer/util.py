# Copyright 2018 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Utilities to download and preprocess the Census data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import pandas as pd
import tensorflow as tf

# Storage directory
DATA_DIR = '/tmp/census_data/'

# Download options.
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
EVAL_FILE = 'adult.test'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

# These are the features in the dataset.
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

# This is the label (target) we want to predict.
_LABEL_COLUMN = 'income_bracket'

# Dataset information: https://archive.ics.uci.edu/ml/datasets/census+income
# fnlwgt: The number of people the census takers believe that observation
# represents. We will be ignoring this variable.
UNUSED_COLUMNS = ['fnlwgt']

# Make model "Fairer"
# More info: https://developers.google.com/machine-learning/fairness-overview/
FAIR_COLUMNS = ['gender']


def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format.

  Args:
    filename: filename to save url to
    url: URL of resource to download
  """
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.gfile.Open(temp_file, 'r') as temp_file_object:
    with tf.gfile.Open(filename, 'w') as file_object:
      for line in temp_file_object:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        file_object.write(line)
  tf.gfile.Remove(temp_file)


def download(data_dir):
  """Downloads census data if it is not already present.

  Args:
    data_dir: directory where we will access/save the census data
  """
  tf.gfile.MakeDirs(data_dir)

  training_file_path = os.path.join(data_dir, TRAINING_FILE)
  if not tf.gfile.Exists(training_file_path):
    _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(data_dir, EVAL_FILE)
  if not tf.gfile.Exists(eval_file_path):
    _download_and_clean_file(eval_file_path, EVAL_URL)


def preprocess(dataframe):
  """Converts categorical features in the data to be numeric.

  Args:
    dataframe: Pandas dataframe with raw data

  Returns:
    Dataframe with preprocessed data
  """
  # Convert integer valued (numeric) columns to floating point
  numeric_columns = dataframe.select_dtypes(['int64']).columns
  dataframe[numeric_columns] = dataframe[numeric_columns].astype('float32')

  # Convert categorical columns to numeric
  cat_columns = dataframe.select_dtypes(['object']).columns
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.astype(
      'category'))
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)
  return dataframe


def standardization(dataframe):
  """Standardizes numeric fields to have mean of 0 and standard deviation of 1.

  Args:
    dataframe: Pandas dataframe with fields to standardize

  Returns:
    Dataframe with standardized fields
  """
  dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
  # Normalize numeric columns.
  for column, dtype in dtypes:
    if dtype == 'float32':
      dataframe[column] -= dataframe[column].mean()
      dataframe[column] /= dataframe[column].std()
  return dataframe


def load_data():
  """Loads data into preprocessed (train_x, train_y, test_x, test_y) dataframes.

  Returns:
    A tuple (train_x, train_y, test_x, test_y), where train_x and test_x are
    Pandas dataframes with features for training and train_y and test_y are
    numpy arrays with the corresponding labels.
  """
  # Download Census dataset: Training and test csv files.
  download(DATA_DIR)

  # Define the full path for training and test files.
  train_file = os.path.join(DATA_DIR, TRAINING_FILE)
  test_file = os.path.join(DATA_DIR, EVAL_FILE)

  # This census data uses the value '?' for fields (column) that are missing
  # data. We use na_values to find ? and set it to NaN values.
  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
  train = pd.read_csv(train_file, names=_CSV_COLUMNS, na_values='?')
  test = pd.read_csv(test_file, names=_CSV_COLUMNS, na_values='?')

  train = preprocess(train)
  test = preprocess(test)

  # Split train and test data with labels.
  # The pop() method will extract (copy) and remove the label column from the
  # dataframe
  train_x, train_y = train, train.pop(_LABEL_COLUMN)
  test_x, test_y = test, test.pop(_LABEL_COLUMN)

  train_x = standardization(train_x)
  test_x = standardization(test_x)

  # Drop unused and biased columns
  train_x = train_x.drop(FAIR_COLUMNS + UNUSED_COLUMNS, axis=1)
  test_x = test_x.drop(FAIR_COLUMNS + UNUSED_COLUMNS, axis=1)

  # Reshape Label for Dataset.
  train_y = np.asarray(train_y).astype('float32').reshape((-1, 1))
  test_y = np.asarray(test_y).astype('float32').reshape((-1, 1))

  return train_x, train_y, test_x, test_y
