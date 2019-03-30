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

import argparse
import logging
import sys

from sklearn import model_selection

from trainer import input_util
from trainer import model


def _train_and_evaluate(estimator, dataset, output_dir):
  """Runs model training and evalation."""

  x_train, y_train, x_val, y_val = dataset

  estimator.fit(x_train, y_train)

  # Note: for now, use `cross_val_score` defaults (i.e. 3-fold)
  scores = model_selection.cross_val_score(estimator, x_val, y_val)

  logging.info(scores)

  # TODO(cezequiel): Write model and eval metrics to `output_dir`.
  _ = output_dir


def run_experiment(flags):
  """Testbed for running model training and evaluation."""

  # Get data for training and evaluation
  dataset = input_util.read_from_bigquery(flags.bq_table)

  # Get model
  estimator = model.get_estimator(flags)

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
    default='INFO',
  )

  return parser.parse_args(argv)


def main():
  flags = _parse_args(sys.argv[1:])
  logging.basicConfig(level=flags.log_level.upper())
  run_experiment(flags)


if __name__ == '__main__':
  main()
