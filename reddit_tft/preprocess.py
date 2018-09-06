# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Reddit Classification Sample Preprocessing Runner."""
import argparse
import datetime
import os
import random
import subprocess
import sys

import path_constants
import reddit

import apache_beam as beam
import tensorflow as tf
from tensorflow_transform import coders
from tensorflow_transform.beam import impl as tft
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.tf_metadata import dataset_metadata


def _default_project():
  get_project = [
      'gcloud', 'config', 'list', 'project', '--format=value(core.project)'
  ]

  with open(os.devnull, 'w') as dev_null:
    return subprocess.check_output(get_project, stderr=dev_null).strip()


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments including program name.
  Returns:
    The parsed arguments as returned by argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on the Reddit model data.')

  parser.add_argument(
      '--project_id', help='The project to which the job will be submitted.')
  parser.add_argument(
      '--cloud', action='store_true', help='Run preprocessing on the cloud.')
  parser.add_argument(
      '--frequency_threshold',
      type=int,
      default=10,
      help=
      'The frequency threshold below which categorical values are ignored.')
  parser.add_argument(
      '--training_data',
      required=True,
      help='Data to analyze and encode as training features.')
  parser.add_argument(
      '--eval_data',
      required=True,
      help='Data to encode as evaluation features.')
  parser.add_argument(
      '--predict_data', help='Data to encode as prediction features.')
  parser.add_argument(
      '--output_dir',
      required=True,
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))
  args, _ = parser.parse_known_args(args=argv[1:])

  if args.cloud and not args.project_id:
    args.project_id = _default_project()

  return args


class _ReadData(beam.PTransform):
  """Wrapper for reading from either CSV files or from BigQuery."""

  def __init__(self, handle, mode=tf.contrib.learn.ModeKeys.TRAIN):
    self._handle = handle
    self._mode = mode

  def expand(self, pvalue):
    if self._handle.endswith('.csv'):
      # The input is CSV file(s).
      schema = reddit.make_input_schema(mode=self._mode)
      csv_coder = reddit.make_csv_coder(schema, mode=self._mode)
      return (pvalue.pipeline
              | 'ReadFromText' >> beam.io.ReadFromText(
                  self._handle,
                  # TODO(b/35653662): Obviate the need for setting this.
                  coder=beam.coders.BytesCoder())
              | 'ParseCSV' >> beam.Map(csv_coder.decode))
    else:
      # The input is BigQuery table name(s).
      query = reddit.make_standard_sql(self._handle, mode=self._mode)
      return (pvalue.pipeline
              | 'ReadFromBigQuery' >> beam.io.Read(
                  beam.io.BigQuerySource(query=query, use_standard_sql=True)))


# TODO: Perhaps use Reshuffle (https://issues.apache.org/jira/browse/BEAM-1872)?
@beam.ptransform_fn
def _Shuffle(pcoll):  # pylint: disable=invalid-name
  return (pcoll
          | 'PairWithRandom' >> beam.Map(lambda x: (random.random(), x))
          | 'GroupByRandom' >> beam.GroupByKey()
          | 'DropRandom' >> beam.FlatMap(lambda (k, vs): vs))


def preprocess(pipeline, training_data, eval_data, predict_data, output_dir,
               frequency_threshold):
  """Run pre-processing step as a pipeline.

  Args:
    pipeline: beam pipeline
    training_data: the name of the table to train on.
    eval_data: the name of the table to evaluate on.
    predict_data: the name of the table to predict on.
    output_dir: file path to where to write all the output files.
    frequency_threshold: frequency threshold to use for categorical values.
  """

  # 1) The schema can be either defined in-memory or read from a configuration
  #    file, in this case we are creating the schema in-memory.
  input_schema = reddit.make_input_schema()

  # 2) Read from BigQuery or from CSV.
  train_data = pipeline | 'ReadTrainingData' >> _ReadData(training_data)
  evaluate_data = pipeline | 'ReadEvalData' >> _ReadData(eval_data)

  input_metadata = dataset_metadata.DatasetMetadata(schema=input_schema)

  _ = (input_metadata
       | 'WriteInputMetadata' >> tft_beam_io.WriteMetadata(
           os.path.join(output_dir, path_constants.RAW_METADATA_DIR),
           pipeline=pipeline))

  preprocessing_fn = reddit.make_preprocessing_fn(frequency_threshold)
  transform_fn = ((train_data, input_metadata)
                  | 'Analyze' >> tft.AnalyzeDataset(preprocessing_fn))

  # WriteTransformFn writes transform_fn and metadata to fixed subdirectories
  # of output_dir, which are given by path_constants.TRANSFORM_FN_DIR and
  # path_constants.TRANSFORMED_METADATA_DIR.
  _ = (transform_fn
       | 'WriteTransformFn' >> tft_beam_io.WriteTransformFn(output_dir))

  @beam.ptransform_fn
  def TransformAndWrite(pcoll, path):  # pylint: disable=invalid-name
    pcoll |= 'Shuffle' >> _Shuffle()  # pylint: disable=no-value-for-parameter
    (dataset, metadata) = (((pcoll, input_metadata), transform_fn)
                           | 'Transform' >> tft.TransformDataset())
    coder = coders.ExampleProtoCoder(metadata.schema)
    _ = (dataset
         | 'SerializeExamples' >> beam.Map(coder.encode)
         | 'WriteExamples' >> beam.io.WriteToTFRecord(
             os.path.join(output_dir, path), file_name_suffix='.tfrecord.gz'))

  _ = train_data | 'TransformAndWriteTraining' >> TransformAndWrite(  # pylint: disable=no-value-for-parameter
      path_constants.TRANSFORMED_TRAIN_DATA_FILE_PREFIX)

  _ = evaluate_data | 'TransformAndWriteEval' >> TransformAndWrite(  # pylint: disable=no-value-for-parameter
      path_constants.TRANSFORMED_EVAL_DATA_FILE_PREFIX)

  # TODO(b/35300113) Remember to eventually also save the statistics.

  if predict_data:
    predict_mode = tf.contrib.learn.ModeKeys.INFER
    predict_schema = reddit.make_input_schema(mode=predict_mode)
    predict_coder = coders.ExampleProtoCoder(predict_schema)

    serialized_examples = (pipeline
                           | 'ReadPredictData' >> _ReadData(
                               predict_data, mode=predict_mode)
                           # TODO(b/35194257) Obviate the need for this explicit
                           # serialization.
                           | 'EncodePredictData' >> beam.Map(
                               predict_coder.encode))
    _ = (serialized_examples
         | 'WritePredictDataAsTFRecord' >> beam.io.WriteToTFRecord(
             os.path.join(output_dir,
                          path_constants.TRANSFORMED_PREDICT_DATA_FILE_PREFIX),
             file_name_suffix='.tfrecord.gz'))
    _ = (serialized_examples
         | 'EncodePredictAsB64Json' >> beam.Map(_encode_as_b64_json)
         | 'WritePredictDataAsText' >> beam.io.WriteToText(
             os.path.join(output_dir,
                          path_constants.TRANSFORMED_PREDICT_DATA_FILE_PREFIX),
             file_name_suffix='.txt'))


def _encode_as_b64_json(serialized_example):
  import base64  # pylint: disable=g-import-not-at-top
  import json  # pylint: disable=g-import-not-at-top
  return json.dumps({'b64': base64.b64encode(serialized_example)})


def main(argv=None):
  """Run Preprocessing as a Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)
  if args.cloud:
    pipeline_name = 'DataflowRunner'
    options = {
        'job_name': ('cloud-ml-sample-reddit-preprocess-{}'.format(
            datetime.datetime.now().strftime('%Y%m%d%H%M%S'))),
        'temp_location':
            os.path.join(args.output_dir, 'tmp'),
        'project':
            args.project_id,
        # TODO(b/35727492) Remove this.
        'max_num_workers':
            250,
        'setup_file':
            os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                'setup.py')),
    }
  else:
    pipeline_name = 'DirectRunner'
    options = {
      'project': args.project_id
    }
  
  pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)

  temp_dir = os.path.join(args.output_dir, 'tmp')
  with beam.Pipeline(pipeline_name, options=pipeline_options) as p:
    with tft.Context(temp_dir=temp_dir):
      preprocess(
          pipeline=p,
          training_data=args.training_data,
          eval_data=args.eval_data,
          predict_data=args.predict_data,
          output_dir=args.output_dir,
          frequency_threshold=args.frequency_threshold)


if __name__ == '__main__':
  main()
