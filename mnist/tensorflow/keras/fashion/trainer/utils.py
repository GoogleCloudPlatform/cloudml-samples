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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
import subprocess


WORKING_DIR = os.getcwd()
FASHION_MNIST_TRAIN = 'train-images-idx3-ubyte.gz'
FASHION_MNIST_TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
FASHION_MNIST_TEST = 't10k-images-idx3-ubyte.gz'
FASHION_MNIST_TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


def download_files_from_gcs(source, destination):
  """Download files from GCS to a WORKING_DIR/.

  Args:
    source: GCS path to the training data
    destination: GCS path to the validation data.

  Returns:
    A list to the local data paths where the data is downloaded.
  """
  local_file_names = [destination]
  gcs_input_paths = [source]

  # Copy raw files from GCS into local path.
  raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name)
    for local_file_name in local_file_names
    ]
  for i, gcs_input_path in enumerate(gcs_input_paths):
    if gcs_input_path:
      subprocess.check_call(
        ['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])

  return raw_local_files_data_paths


def _load_data(path, destination):
  """Verifies if file is in Google Cloud.

  Args:
    path: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')
    destination: (str) The filename to save as on local disk.

  Returns:
    A filename
  """
  if path.startswith('gs://'):
    download_files_from_gcs(path, destination=destination)
    return destination
  return path


def prepare_data(train_file, train_labels_file, test_file, test_labels_file):
  """Loads MNIST Fashion files.

    License:
        The copyright for Fashion-MNIST is held by Zalando SE.
        Fashion-MNIST is licensed under the [MIT license](
        https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

  Args:
    train_file: (str) Location where training data file is located.
    train_labels_file: (str) Location where training labels file is located.
    test_file: (str) Location where test data file is located.
    test_labels_file: (str) Location where test labels file is located.

  Returns:
    A tuple of training and test data.
  """
  train_labels_file = _load_data(train_labels_file, FASHION_MNIST_TRAIN_LABELS)
  train_file = _load_data(train_file, FASHION_MNIST_TRAIN)
  test_labels_file = _load_data(test_labels_file, FASHION_MNIST_TEST_LABELS)
  test_file = _load_data(test_file, FASHION_MNIST_TEST)

  with gzip.open(train_labels_file, 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(train_file, 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(test_labels_file, 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(test_file, 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)
