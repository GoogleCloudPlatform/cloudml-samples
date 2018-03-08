#!/usr/bin/env python

# Copyright 2018 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import print_function

import argparse
import os
import subprocess
import sys
import tempfile

from datetime import datetime

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft

from apache_beam.io import filebasedsource
from apache_beam.options import pipeline_options
from tensorflow_transform.tf_metadata import dataset_schema

import data_extractor
import preprocess
import pubchem
import trainer


DEFAULT_WORK_DIR = os.path.join(
    tempfile.gettempdir(), 'cloudml', 'molecules')


INPUT_SCHEMA = {
    # Features
    'TotalC': dataset_schema.ColumnSchema(
        tf.int64, [], dataset_schema.FixedColumnRepresentation()),
    'TotalH': dataset_schema.ColumnSchema(
        tf.int64, [], dataset_schema.FixedColumnRepresentation()),
    'TotalO': dataset_schema.ColumnSchema(
        tf.int64, [], dataset_schema.FixedColumnRepresentation()),
    'TotalN': dataset_schema.ColumnSchema(
        tf.int64, [], dataset_schema.FixedColumnRepresentation()),

    # Labels
    'Energy': dataset_schema.ColumnSchema(
        tf.float32, [], dataset_schema.FixedColumnRepresentation()),
}

LABELS = ['Energy']


class PubChemSource(filebasedsource.FileBasedSource):
  """This is a custom Source to parse through PubChem (.sdf) files.

  It extends a FileBasedSource, but it won't use the TextSource implementation
  since every record in a PubChem file is multiple lines long.
  """

  def read_records(self, file_name, range_tracker):
    """This yields a dictionary with the sections of the file that we're
    interested in. The `range_tracker` allows us to mark the position where
    possibly another worker left, so we make sure to start from there.
    """
    with self.open_file(file_name) as f:
      f.seek(range_tracker.start_position() or 0)
      while range_tracker.try_claim(f.tell()):
        for elem in pubchem.parse_molecules(f):
          yield elem


class PubChemToInputData(beam.PTransform):
  """The preprocessing pipeline (element-wise transformations).

  We create a `PTransform` class. This `PTransform` is a bundle of
  transformations that cann be applied to any other pipeline as a step.

  We'll do all the preprocessing needed here. Due to the nature of
  `PTransform`s, we can only do element-wise transformations here. Anything that
  requires a full-pass of the data has to be done with tf.Transform.
  """
  def __init__(self, data_files_pattern):
    super(PubChemToInputData, self).__init__()
    self.data_files_pattern = data_files_pattern

  def expand(self, p):
    # Helper functions
    def count_atom_type(elem, atom_type):
      return sum(int(atom['type'] == atom_type) for atom in elem['atoms'])

    def format_molecule(elem):
      label = float(elem['energy'])
      return {
        'TotalC': count_atom_type(elem, 'C'),
        'TotalH': count_atom_type(elem, 'H'),
        'TotalO': count_atom_type(elem, 'O'),
        'TotalN': count_atom_type(elem, 'N'),
        'Energy': label,
      }

    # Return the preprocessing pipeline. In this case we're reading the PubChem
    # files, but the source could be any Apache Beam source.
    return (
        p
        | 'ReadPubChem' >> beam.io.Read(PubChemSource(self.data_files_pattern))
        | 'FormatMolecules' >> beam.Map(format_molecule))


def normalize_inputs(inputs):
  """Preprocessing function for tf.Transform (full-pass transformations).

  Here we will do any
  preprocessing that requires a full-pass of the dataset. It takes as inputs the
  preprocessed data from the `PTransform` we specify, in this case
  `PubChemToInputData`.

  Common operations might be normalizing values to 0-1, getting the minimum or
  maximum value of a certain field, creating a vocabulary for a string field.

  There are two main types of transformations supported by tf.Transform, for
  more information, check the following modules:
    - analyzers: tensorflow_transform.analyzers.py
    - mappers:   tensorflow_transform.mappers.py
  """
  return {
      # Scale the input features for normalization
      'NormalizedC': tft.scale_to_0_1(inputs['TotalC']),
      'NormalizedH': tft.scale_to_0_1(inputs['TotalH']),
      'NormalizedO': tft.scale_to_0_1(inputs['TotalO']),
      'NormalizedN': tft.scale_to_0_1(inputs['TotalN']),

      # Do not scale the label since we want the absolute number for prediction
      'Energy': inputs['Energy'],
  }


def make_estimator(run_config):
  """Here we create an estimator.

  In this case we're using a canned estimator.
  """
  estimator = tf.estimator.DNNRegressor(
      feature_columns=[
          tf.feature_column.numeric_column('NormalizedC', dtype=tf.float32),
          tf.feature_column.numeric_column('NormalizedH', dtype=tf.float32),
          tf.feature_column.numeric_column('NormalizedO', dtype=tf.float32),
          tf.feature_column.numeric_column('NormalizedN', dtype=tf.float32),
      ],
      hidden_units=[128, 64, 32],
      dropout=0.2,
      config=run_config)
  return estimator


def timestamp():
  """Wrapper to create a timestamp string."""
  return datetime.now().strftime('%Y%m%d%H%M%S')


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # General settings
  parser.add_argument('--model-name',
                      type=str,
                      default='molecules',
                      help='Model name for the Exporter, will be used in the '
                           'export path')
  parser.add_argument('--work-dir',
                      type=str,
                      default=DEFAULT_WORK_DIR,
                      help='Working directory for the script, '
                           'can be in Google Cloud Storage')

  # Data extraction
  parser.add_argument('--total-data-files',
                      type=int,
                      default=5,
                      help='Total number of data files to use, '
                           'set to `-1` to use all data files. '
                           'Each data file contains 25,000 molecules')
  parser.add_argument('--eval-percent',
                      type=float,
                      default=20.0,
                      help='Percentage of the dataset to use for the '
                           'evaluation set (0-100)')

  # Training settings
  parser.add_argument('--train-max-steps',
                      type=int,
                      default=1000,
                      help='Number of steps to train the model')
  parser.add_argument('--batch-size',
                      type=int,
                      default=64,
                      help='Batch size for training and evaluation')

  # Google Cloud settings
  parser.add_argument('--cloud',
                      default=False,
                      action='store_true',
                      help='Run in Google Cloud. The preprocessing will be '
                           'done in Dataflow and the ML model training will '
                           'be done in Cloud ML Engine.')
  parser.add_argument('--project',
                      type=str,
                      default=None,
                      help='The Google Cloud Project ID when running with '
                           'the `--cloud` option')
  parser.add_argument('--region',
                      type=str,
                      default='us-central1',
                      help='The Google Compute Engine region for the Dataflow '
                           'and Cloud ML Engine jobs')

  args, pipeline_args = parser.parse_known_args()

  # 1) Input validation
  if args.total_data_files < 1 and args.total_data_files != -1:
    print('Error: --total-data-files must be >= 1 or -1 to use all')
    sys.exit(1)

  if args.total_data_files == -1:
    args.total_data_files = None

  if args.eval_percent <= 0 or args.eval_percent >= 100:
    print('Error: --eval-percent must be in the range (0-100)')
    sys.exit(1)

  beam_options = None
  if args.cloud:
    if args.project is None:
      print('error: argument --project is required with --cloud')
      sys.exit(1)

    if not args.work_dir.startswith('gs://'):
      print('error: the --work-dir location must be in Google Cloud Storage '
            'when running with --cloud')
      sys.exit(1)

    setup_file = os.path.join(os.path.dirname(__file__), 'setup.py')
    beam_args = {
        'job_name': 'cloudml-molecules-preprocess-{}'.format(timestamp()),
        'runner': 'DataflowRunner',
        'project': args.project,
        'temp_location': os.path.join(args.work_dir, 'beam_temp'),
        'save_main_session': True,
        'setup_file': os.path.abspath(setup_file),
        'region': args.region,
    }
    beam_options = pipeline_options.PipelineOptions(flags=[], **beam_args)

  print('Project: {}'.format(args.project))
  print('GOOGLE_APPLICATION_CREDENTIALS: {}'.format(
      os.environ['GOOGLE_APPLICATION_CREDENTIALS']))
  print()

  # Have a unique directory for every run
  work_dir = os.path.join(args.work_dir, timestamp())
  print('Working directory: {}'.format(work_dir))
  print()

  # 2) Data extraction
  print('Data extraction')
  raw_data_dir = os.path.join(work_dir, 'raw_data')
  raw_data_files_pattern = data_extractor.extract(
      args.total_data_files, raw_data_dir)
  print('Data extraction done')
  print()

  # 4) Export the TrainerData
  print('Exporting TrainerData')
  model_dir = os.path.join(work_dir, 'model')

  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=model_dir)
  estimator = make_estimator(run_config)

  trainer_data = trainer.TrainerData(
      estimator, train_input_fn, eval_input_fn, serving_input_fn)

  trainer_data_filename = os.path.join(work_dir, 'TrainerData')
  trainer.dump(trainer_data, trainer_data_filename)
  print('TrainerData: {}'.format(trainer_data))

  print('Success! :D')
