#!/usr/bin/env python
#
# Copyright 2018 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import argparse
import dill as pickle
import os
import random
import tempfile

import pubchem

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform.beam.impl as beam_impl

from apache_beam.io import tfrecordio
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


class PreprocessData(object):
  def __init__(
      self,
      input_feature_spec,
      labels,
      train_files_pattern,
      eval_files_pattern):

    self.labels = labels
    self.input_feature_spec = input_feature_spec
    self.train_files_pattern = train_files_pattern
    self.eval_files_pattern = eval_files_pattern


def dump(obj, filename):
  """ Wrapper to dump an object to a file."""
  with tf.gfile.Open(filename, 'wb') as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
  """ Wrapper to load an object from a file."""
  with tf.gfile.Open(filename, 'rb') as f:
    return pickle.load(f)


class ValidateInputData(beam.DoFn):
  """This DoFn validates that every element matches the metadata given."""
  def __init__(self, input_metadata):
    super(ValidateInputData, self).__init__()
    self.schema_keys = set(input_metadata.schema.column_schemas.keys())

  def process(self, elem):
    if not isinstance(elem, dict):
      raise ValueError(
          'Element must be a dict(str, value). '
          'Given: {} {}'.format(elem, type(elem)))
    elem_keys = set(elem.keys())
    if not self.schema_keys.issubset(elem_keys):
      raise ValueError(
          "Element keys are missing from schema keys. "
          'Given: {}; Schema: {}'.format(
              list(elem_keys), list(self.schema_keys)))
    yield elem


def run(
    input_schema,
    labels,
    preprocessing,
    full_pass_preprocessing_fn=None,
    eval_percent=20.0,
    beam_options=None,
    work_dir=None):
  """Runs the whole preprocessing step.

  This runs the preprocessing PTransform, validates that the data conforms to
  the schema provided, does the full-pass preprocessing step and generates the
  input functions needed to train and evaluate the TensorFlow model.
  """

  # Populate optional arguments
  if not full_pass_preprocessing_fn:
    full_pass_preprocessing_fn = lambda inputs: inputs

  # Type checking
  if not isinstance(labels, list):
    raise ValueError(
        '`labels` must be list(str). '
        'Given: {} {}'.format(labels, type(labels)))

  if not isinstance(preprocessing, beam.PTransform):
    raise ValueError(
        '`preprocessing` must be {}. '
        'Given: {} {}'.format(beam.PTransform,
            preprocessing, type(preprocessing)))

  if not callable(full_pass_preprocessing_fn):
    raise ValueError(
        '`full_pass_preprocessing_fn` must be callable. '
        'Given: {} {}'.format(full_pass_preprocessing_fn,
            type(full_pass_preprocessing_fn)))

  if beam_options and not isinstance(beam_options, PipelineOptions):
    raise ValueError(
        '`beam_options` must be {}. '
        'Given: {} {}'.format(PipelineOptions,
            beam_options, type(beam_options)))

  if not work_dir:
    work_dir = tempfile.mkdtemp(prefix='tensorflow-preprocessing')

  tft_temp_dir = os.path.join(work_dir, 'tft-temp')
  train_dataset_dir = os.path.join(work_dir, 'train-dataset')
  eval_dataset_dir = os.path.join(work_dir, 'eval-dataset')

  transform_fn_dir = os.path.join(work_dir, transform_fn_io.TRANSFORM_FN_DIR)
  if tf.gfile.Exists(transform_fn_dir):
    tf.gfile.DeleteRecursively(transform_fn_dir)

  input_metadata = dataset_metadata.DatasetMetadata(
      dataset_schema.Schema(input_schema))

  # Build and run a Beam Pipeline
  with beam.Pipeline(options=beam_options) as p, \
       beam_impl.Context(temp_dir=tft_temp_dir):

    # Transform and validate the input data matches the input schema
    dataset = (
        p
        | 'Preprocessing' >> preprocessing
        | 'Validate inputs' >> beam.ParDo(ValidateInputData(input_metadata)))

    # Apply the tf.Transform preprocessing_fn
    dataset_and_metadata, transform_fn = (
        (dataset, input_metadata)
        | 'Full-pass preprocessing' >> beam_impl.AnalyzeAndTransformDataset(
            full_pass_preprocessing_fn))

    dataset, metadata = dataset_and_metadata

    # Split the dataset into a training set and an evaluation set
    assert 0 < eval_percent < 100, 'eval_percent must in the range (0-100)'
    train_dataset, eval_dataset = (
        dataset
        | 'Split dataset' >> beam.Partition(
            lambda elem, _: int(random.uniform(0, 100) < eval_percent), 2))

    # Write the datasets as TFRecords
    coder = example_proto_coder.ExampleProtoCoder(metadata.schema)

    train_dataset_prefix = os.path.join(train_dataset_dir, 'part')
    _ = (
        train_dataset
        | 'Write train dataset' >> tfrecordio.WriteToTFRecord(
            train_dataset_prefix, coder))

    eval_dataset_prefix = os.path.join(eval_dataset_dir, 'part')
    _ = (
        eval_dataset
        | 'Write eval dataset' >> tfrecordio.WriteToTFRecord(
            eval_dataset_prefix, coder))

    # Write the transform_fn
    _ = (
        transform_fn
        | 'Write transformFn' >> transform_fn_io.WriteTransformFn(work_dir))

  return PreprocessData(
      input_metadata.schema.as_feature_spec(),
      labels,
      train_dataset_prefix + '*',
      eval_dataset_prefix + '*')


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--work-dir',
      type=str,
      default=os.path.join(
        tempfile.gettempdir(), 'cloudml-samples', 'molecules'),
      help='Directory for staging and working files. '
           'This can be a Google Cloud Storage path.')

  args, pipeline_args = parser.parse_known_args()

  beam_options = PipelineOptions(pipeline_args)
  beam_options.view_as(SetupOptions).save_main_session = True

  data_files_pattern = os.path.join(args.work_dir, 'data', '*.sdf')
  preprocess_data = run(
      pubchem.INPUT_SCHEMA,
      pubchem.LABELS,
      pubchem.SimpleFeatureExtraction(
          beam.io.Read(pubchem.ParseSDF(data_files_pattern))),
      full_pass_preprocessing_fn=pubchem.normalize_inputs,
      beam_options=beam_options,
      work_dir=args.work_dir)

  dump(preprocess_data, os.path.join(args.work_dir, 'PreprocessData'))
