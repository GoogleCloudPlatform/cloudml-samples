# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import re
import numpy as np

from google.cloud import storage

BOSTON_FILE = 'boston_housing.npz'


def _download_from_gcs(source, destination):
  """Downloads data from Google Cloud Storage into current local folder.

  Destination MUST be filename ONLY, doesn't support folders.
  (e.g. 'file.csv', NOT 'folder/file.csv')

  Args:
    source: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')
    destination: (str) The filename to save as on local disk.

  Returns:
    Nothing, downloads file to local disk.
  """
  search = re.search('gs://(.*?)/(.*)', source)
  bucket_name = search.group(1)
  blob_name = search.group(2)
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  bucket.blob(blob_name).download_to_filename(destination)


def load_data(path='boston_housing.npz', test_split=0.2, seed=113):
  """Loads the Boston Housing dataset.

  Args:
    path: path where to cache the dataset locally (relative to
        ~/.keras/datasets).
    test_split: fraction of the data to reserve as test set.
    seed: Random seed for shuffling the data before computing the test split.

  Returns:
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  Raises:
    ValueError: No dataset file defined.
  """
  assert 0 <= test_split < 1
  if not path:
    raise ValueError('No dataset file defined')

  if path.startswith('gs://'):
    _download_from_gcs(path, destination=BOSTON_FILE)
    path = BOSTON_FILE

  with np.load(path) as f:
    x = f['x']
    y = f['y']

  np.random.seed(seed)
  indices = np.arange(len(x))
  np.random.shuffle(indices)
  x = x[indices]
  y = y[indices]

  x_train = np.array(x[:int(len(x) * (1 - test_split))])
  y_train = np.array(y[:int(len(x) * (1 - test_split))])
  x_test = np.array(x[int(len(x) * (1 - test_split)):])
  y_test = np.array(y[int(len(x) * (1 - test_split)):])
  return (x_train, y_train), (x_test, y_test)


def normalize_data(train_data, test_data):
  """Normalize features with different scales and ranges.

  Subtract the mean of the feature and divide by the standard deviation.
  Test data is *not* used when calculating the mean and std.

  Args:
    train_data: (numpy.darray) Training data.
    test_data: (numpy.darray) Testing data.

  Returns:
    A tuple of training and test data.
  """
  mean = train_data.mean(axis=0)
  std = train_data.std(axis=0)
  train_data = (train_data - mean) / std
  test_data = (test_data - mean) / std
  return train_data, test_data