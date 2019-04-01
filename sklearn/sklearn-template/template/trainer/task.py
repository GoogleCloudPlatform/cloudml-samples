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

"""Executes model training and evaluation."""

import os
import time
import argparse
import logging
import sys
from urllib.parse import urlparse

from sklearn import model_selection
from sklearn.externals import joblib

import numpy as np

import hypertune

from google.cloud import storage

from trainer.input_util import read_from_bigquery
from trainer.model import get_estimator
from trainer.metadata import MODEL_FILE_NAME_PREFIX
from trainer.metadata import METRIC_FILE_NAME_PREFIX
from trainer.metadata import DUMP_FILE_NAME_SUFFIX


def _upload_to_gcs(local_path, gcs_path):
  """
  Upload local file to Google Cloud Storage

  Args:
    local_path: (string) Local file
    gcs_path: (string) Google Cloud Storage destination

  Returns:
    None
  """
  storage_client = storage.Client()
  parse_result = urlparse(gcs_path)

  # Parse bucket name
  gcs_path = parse_result.path
  bucket_name = parse_result.hostname
  bucket = storage_client.get_bucket(bucket_name)

  blob_path = gcs_path[1:] if gcs_path[0] == '/' else gcs_path
  blob = bucket.blob(blob_path)

  if blob.exists():
    blob.delete()
  blob.upload_from_filename(local_path)


def _dump_object(object_to_dump, output_path):
  """
  Pickle the object and save to the output_path

  Args:
    object_to_dump: Python object to be pickled
    output_path: (string) output path which can be Google Cloud Storage

  Returns:
    None
  """
  gcs_path = None
  parse_result = urlparse(output_path)

  if parse_result.scheme == 'gs':
    file_name = os.path.basename(parse_result.path)
    gcs_path = output_path
  else:
    file_name = output_path

  with open(file_name, 'wb') as wf:
    joblib.dump(object_to_dump, wf)

  if gcs_path:
    _upload_to_gcs(file_name, gcs_path)


def _train_and_evaluate(estimator, dataset, output_dir):
  """Runs model training and evalation."""
  # TODO: How to recover the dumped model with corresponding setting of hyperparameters ?
  x_train, y_train, x_val, y_val = dataset

  estimator.fit(x_train, y_train)

  # Note: for now, use `cross_val_score` defaults (i.e. 3-fold)
  scores = model_selection.cross_val_score(estimator, x_val, y_val)

  logging.info(scores)

  # The default name of the metric is training/hptuning/metric.
  # We recommend that you assign a custom name. The only functional difference is that
  # if you use a custom name, you must set the hyperparameterMetricTag value in the
  # HyperparameterSpec object in your job request to match your chosen name.
  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='my_metric_tag',
    metric_value=np.mean(scores),
    global_step=1000)

  timestamp = str(int(time.time()))
  trial_id = str(hpt.trial_id)

  # Export to the folder of output_dir/trial_id/FILE_NAME_PREFIX_timestampFILE_NAME_SUFFIX
  model_output_path = os.path.join(output_dir, trial_id,
                                   (MODEL_FILE_NAME_PREFIX
                                    + '_' + timestamp
                                    + DUMP_FILE_NAME_SUFFIX))

  metric_output_path = os.path.join(output_dir, trial_id,
                                    (METRIC_FILE_NAME_PREFIX
                                     + '_' + timestamp
                                     + DUMP_FILE_NAME_SUFFIX))

  _dump_object(estimator, model_output_path)
  _dump_object(scores, metric_output_path)


def run_experiment(flags):
  """Testbed for running model training and evaluation."""
  # Get data for training and evaluation
  dataset = read_from_bigquery(flags.bq_table)

  # Get model
  estimator = get_estimator(flags)

  # Run training and evaluation
  _train_and_evaluate(estimator, dataset, flags.job_dir)


def _parse_args(argv):
  """Parses command-line arguments."""

  parser = argparse.ArgumentParser()

  # TODO(cezequiel): Change to read from BigQuery table instead.
  parser.add_argument(
    '--bq_table',
    help='Bigquery table containing input dataset.',
    required=True,
  )

  parser.add_argument(
    '--job-dir',
    help='Output directory for exporting model and other metadata.',
    required=True,
  )

  parser.add_argument(
    '--log_level',
    help='Logging level.',
    choices=[
      'DEBUG',
      'ERROR',
      'FATAL',
      'INFO',
      'WARN'
    ],
    default='INFO',
  )

  parser.add_argument(
    '--n_estimator',
    help='Number of trees in the forest.',
    default=10,
    type=int
  )

  parser.add_argument(
    '--max_depth',
    help='The maximum depth of the tree.',
    type=int,
    default=None
  )

  parser.add_argument(
    '--min_samples_leaf',
    help='The minimum number of samples required to be at a leaf node.',
    default=1,
    type=int
  )

  parser.add_argument(
    '--criterion',
    help='The function to measure the quality of a split.',
    choices=[
      'gini',
      'entropy',
    ],
    default='gini'
  )

  return parser.parse_args(argv)


def main():
  flags = _parse_args(sys.argv[1:])
  logging.basicConfig(level=flags.log_level.upper())
  run_experiment(flags)


if __name__ == '__main__':
  main()
