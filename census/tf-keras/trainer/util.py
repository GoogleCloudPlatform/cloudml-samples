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
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

# Storage directory
DATA_DIR = os.path.join(tempfile.gettempdir(), 'census_data')

# Download options.
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
EVAL_FILE = 'adult.test'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

# These are the features in the dataset.
# Dataset information: https://archive.ics.uci.edu/ml/datasets/census+income
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

# This is the label (target) we want to predict.
_LABEL_COLUMN = 'income_bracket'

# These are columns we will not use as features for training. There are many
# reasons not to use certain attributes of data for training. Perhaps their
# values are noisy or inconsistent, or perhaps they encode bias that we do not
# want our trained model to learn. For a deep dive into the features of this
# Census dataset and the challenges they pose, see the Introduction to ML
# Fairness notebook: https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/intro_to_fairness.ipynb
UNUSED_COLUMNS = ['fnlwgt', 'gender']


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
  
  return training_file_path, eval_file_path  


def preprocess(dataframe):
  """Converts categorical features to numeric. Removes unused columns.

  Args:
    dataframe: Pandas dataframe with raw data

  Returns:
    Dataframe with preprocessed data
  """
  dataframe = dataframe.drop(columns=UNUSED_COLUMNS)

  # Convert integer valued (numeric) columns to floating point
  numeric_columns = dataframe.select_dtypes(['int64']).columns
  dataframe[numeric_columns] = dataframe[numeric_columns].astype('float32')

  # Convert categorical columns to numeric
  cat_columns = dataframe.select_dtypes(['object']).columns
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.astype(
      'category'))
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)
  return dataframe


def preprocess_csv(csv_filename):
  """Loads a CSV into a dataframe and preprocesses it for our model.
  
  Can be used to load training/eval data, or test data used for prediction.

  Args:
    csv_filename: Path to a CSV file to load with Pandas and preprocess

  Returns:
    Dataframe with preprocessed data
  """
  # This census data uses the value '?' for fields (column) that are missing
  # data. We use na_values to find ? and set it to NaN values.
  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
  dataframe = pd.read_csv(csv_filename, names=_CSV_COLUMNS, na_values='?')
  return preprocess(dataframe)


def load_data():
  """Loads data into preprocessed (train_x, train_y, eval_y, eval_y) dataframes.

  Returns:
    A tuple (train_x, train_y, eval_x, eval_y), where train_x and eval_x are
    Pandas dataframes with features for training and train_y and eval_y are
    numpy arrays with the corresponding labels.
  """
  # Download Census dataset: Training and eval csv files.
  training_file_path, eval_file_path = download(DATA_DIR)

  train_df = preprocess_csv(training_file_path)
  eval_df = preprocess_csv(eval_file_path)

  # Split train and eval data with labels.
  # The pop() method will extract (copy) and remove the label column from the
  # dataframe
  train_x, train_y = train_df, train_df.pop(_LABEL_COLUMN)
  eval_x, eval_y = eval_df, eval_df.pop(_LABEL_COLUMN)

  # Reshape Label for Dataset.
  train_y = np.asarray(train_y).astype('float32').reshape((-1, 1))
  eval_y = np.asarray(eval_y).astype('float32').reshape((-1, 1))

  return train_x, train_y, eval_x, eval_y
