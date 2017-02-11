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
"""Criteo Classification Sample Preprocessing Runner."""
import argparse
import datetime
import os
import subprocess
import sys

import apache_beam as beam
import trainer.model as criteo

import google.cloud.ml as ml
import google.cloud.ml.io as io


def _default_project():
  get_project = ['gcloud', 'config', 'list', 'project',
                 '--format=value(core.project)']

  with open(os.devnull, 'w') as dev_null:
    return subprocess.check_output(get_project, stderr=dev_null).strip()


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments including program name.
  """
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on the Criteo model data.')

  parser.add_argument(
      '--project_id', help='The project to which the job will be submitted.')
  parser.add_argument(
      '--cloud', action='store_true', help='Run preprocessing on the cloud.')
  parser.add_argument(
      '--frequency_threshold',
      type=int,
      default=100,
      help='The frequency threshold below which categorical values are ignored.'
  )
  parser.add_argument(
      '--training_data',
      required=True,
      help='Data to analyze and encode as training features.')
  parser.add_argument(
      '--eval_data', required=True,
      help='Data to encode as evaluation features.')
  parser.add_argument(
      '--metadata_file_name',
      default='metadata.json',
      help='Name for the metadata file, one of metadata.{json|yaml}.')
  parser.add_argument(
      '--output_dir',
      default=None,
      required=True,
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))
  args, _ = parser.parse_known_args(args=argv[1:])

  if args.cloud and not args.project_id:
    args.project_id = _default_project()

  return args


def preprocess(pipeline, training_data, eval_data, output_dir,
               frequency_threshold, metadata_file_name):
  """Run pre-processing step as a pipeline.

  Args:
    pipeline: beam pipeline
    training_data: file paths to input csv files.
    eval_data: file paths to input csv files.
    output_dir: file path to where to write all the output files.
    frequency_threshold: frequency threshold to use for categorical values.
    metadata_file_name: one of metadata.{json|yaml}.
  """
  feature_set, csv_columns = criteo.criteo_features(
      frequency_threshold=frequency_threshold)

  coder = io.CsvCoder.from_feature_set(feature_set, csv_columns, delimiter='\t')

  train = (
      pipeline
      | 'ReadTrainingData'
      >> beam.io.ReadFromText(
          training_data, strip_trailing_newlines=True, coder=coder))

  evaluate = (
      pipeline
      | 'ReadEvalData'
      >> beam.io.ReadFromText(
          eval_data, strip_trailing_newlines=True, coder=coder))

  # TODO(b/32726166) Update input_format and format_metadata to read from these
  # values directly from the coder.
  (metadata, train_features, evaluate_features) = (
      (train, evaluate)
      | 'Preprocess' >> ml.Preprocess(
          feature_set,
          input_format='csv',
          format_metadata={'headers': csv_columns,
                           'delimiter': '\t'}))

  # Writes metadata.json, features_train, features_eval, and features_eval files
  # pylint: disable=expression-not-assigned
  (metadata
   | 'SaveMetadata'
   >> io.SaveMetadata(os.path.join(output_dir, metadata_file_name)))
  (train_features
   | 'WriteTraining'
   >> io.SaveFeatures(os.path.join(output_dir, 'features_train')))
  (evaluate_features
   | 'WriteEval'
   >> io.SaveFeatures(os.path.join(output_dir, 'features_eval')))


def main(argv=None):
  """Run Preprocessing as a Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)

  if args.cloud:
    options = {
        'staging_location': os.path.join(args.output_dir, 'tmp', 'staging'),
        'temp_location': os.path.join(args.output_dir, 'tmp', 'staging'),
        'job_name': ('cloud-ml-sample-criteo-preprocess-{}'.format(
            datetime.datetime.now().strftime('%Y%m%d%H%M%S'))),
        'project':
            args.project_id,
        'extra_packages': [ml.sdk_location],
        'autoscaling_algorithm':
            'THROUGHPUT_BASED',
        'worker_machine_type':
            'n1-standard-4',
        # TODO(andreasst): remove machine_type once no longer needed
        'machine_type':
            'n1-standard-4',
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    pipeline = beam.Pipeline('DataflowRunner', options=opts)
  else:
    pipeline = beam.Pipeline('DirectRunner')

  preprocess(
      pipeline=pipeline,
      training_data=args.training_data,
      eval_data=args.eval_data,
      output_dir=args.output_dir,
      frequency_threshold=args.frequency_threshold,
      metadata_file_name=args.metadata_file_name)

  pipeline.run().wait_until_finish()


if __name__ == '__main__':
  main()
