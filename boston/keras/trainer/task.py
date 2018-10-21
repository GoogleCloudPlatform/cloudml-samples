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

import argparse
import os
import re

from . import model

import numpy as np
import tensorflow as tf

from tensorflow.contrib.training.python.training import hparam
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


def train_and_evaluate(hparams):
  """Helper function: Trains and evaluate model.

  Args:
    hparams: (dict) Command line parameters passed from task.py
  """
  # Load data.
  (train_data,
   train_labels), (test_data,
                   test_labels) = load_data(path=hparams.dataset_file)

  # Shuffle data.
  order = np.argsort(np.random.random(train_labels.shape))
  train_data = train_data[order]
  train_labels = train_labels[order]

  # Normalize features with different scales and ranges.
  train_data, test_data = normalize_data(train_data, test_data)

  # Running configuration.
  run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)

  # Create TrainSpec.
  train_steps = hparams.num_epochs * len(train_data) / hparams.batch_size
  train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: model.input_fn(
          train_data,
          train_labels,
          hparams.batch_size,
          mode=tf.estimator.ModeKeys.TRAIN),
      max_steps=train_steps)

  # Create EvalSpec.
  exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: model.input_fn(
          test_data,
          test_labels,
          hparams.batch_size,
          mode=tf.estimator.ModeKeys.EVAL),
      steps=None,
      exporters=[exporter],
      start_delay_secs=10,
      throttle_secs=10)

  # Create estimator.
  estimator = model.keras_estimator(
      model_dir=hparams.job_dir,
      config=run_config,
      learning_rate=hparams.learning_rate,
      num_features=train_data.shape[1])

  # Start training.
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='GCS location to write checkpoints and export models')
  parser.add_argument(
      '--dataset-file',
      type=str,
      required=True,
      help='Dataset file local or GCS')
  parser.add_argument(
      '--test-split',
      type=float,
      default=0.2,
      help='Split between training and test, default=0.2')
  parser.add_argument(
      '--num-epochs',
      type=float,
      default=500,
      help='number of times to go through the data, default=500')
  parser.add_argument(
      '--batch-size',
      type=int,
      default=128,
      help='number of records to read during each training step, default=128')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=.001,
      help='learning rate for gradient descent, default=.001')
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO',
  )

  args = parser.parse_args()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  hparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(hparams)
