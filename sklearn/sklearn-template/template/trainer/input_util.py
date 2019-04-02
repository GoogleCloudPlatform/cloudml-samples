# Copyright 2019 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Data ingest operations."""

import pandas as pd
from sklearn import model_selection
from google.cloud import storage

from trainer.metadata import BASE_QUERY
from trainer.metadata import LABEL


def _feature_label_split(data_df, label_column):
  """Split the DataFrame into two DataFrames including all the features, and label separately

  Args:
    data_df: (pandas.DataFrame) DataFrame the splitting to be performed on
    label_column: (string) name of the label column

  Returns:
    A Tuple of (pandas.DataFrame, pandas.Series)
  """

  return data_df.loc[:, data_df.columns != label_column], data_df[label_column]


def _data_train_test_split(data_df):
  """Split the DataFrame two subsets for training and testing

  Args:
    data_df: (pandas.DataFrame) DataFrame the splitting to be performed on

  Returns:
    A Tuple of (pandas.DataFrame, pandas.Series, pandas.DataFrame, pandas.Series)
  """

  label_column = LABEL
  train, val = model_selection.train_test_split(data_df)
  x_train, y_train = _feature_label_split(train, label_column)
  x_val, y_val = _feature_label_split(val, label_column)
  return x_train, y_train, x_val, y_val


def read_from_bigquery(full_table_path, project_id=None):
  """Read training data from BigQuery given full path of BigQuery table,
  and split into train and validation.

  Args:
    full_table_path: (string) full path of the table containing training data in the
          format of [project_id.dataset_name.table_name]
    project_id: (string, Optional) Google BigQuery Account project ID

  Returns:
    A Tuple of (pandas.DataFrame, pandas.Series, pandas.DataFrame, pandas.Series)
  """

  query = BASE_QUERY.format(Table=full_table_path)

  # Use "application default credentials"
  # Use SQL syntax dialect
  data_df = pd.read_gbq(query, project_id=project_id, dialect='standard')

  return _data_train_test_split(data_df)


def read_from_gcs(file_pattern, bucket_name):
  """Read training data from Google Cloud Storage given the path pattern, and split into train and validation.

  Args:
    file_pattern: (string) pattern of the files containing training data. For example:
          [folder_name/prefix]
    bucket_name: (string) Google Cloud Storage Bucket Name

  Returns:
    A Tuple of (pandas.DataFrame, pandas.Series, pandas.DataFrame, pandas.Series)
  """

  # TODO(luoshixin): Figure out a way to handle the header: metadata.py or data files themself ?
  # Download the files to local /tmp/ foler
  temp_folder = '/tmp/'
  df_list = []

  # Use "application default credentials"
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)

  # Find all the files match the provided pattern
  blobs = bucket.list_blobs(prefix=file_pattern)

  for blob in blobs:
    file_path = temp_folder + blob.name.split('/')[-1]
    blob.download_to_filename(file_path)
    # Assume there is no header
    df_list.append(pd.read_csv(file_path, header=None))
    # TODO: Can remove after download to save space

  data_df = pd.concat(df_list)

  return _data_train_test_split(data_df)


def read_from_bigquery_dump(table_name):
  # TODO(cezequiel): Remove when done using sns dataset
  import seaborn as sns
  data_df = sns.load_dataset('titanic')

  return _data_train_test_split(data_df)
