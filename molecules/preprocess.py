#!/usr/bin/env python

# Copyright 2019 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This tool preprocesses and extracts features from SDF files.

from __future__ import absolute_import

import argparse
import dill as pickle
import logging
import os
import random
import tempfile

import pubchem

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform.beam.impl as beam_impl

from apache_beam.io import tfrecordio
from apache_beam.options.pipeline_options import PipelineOptions
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
  def __init__(self, feature_spec):
    super(ValidateInputData, self).__init__()
    self.feature_names = set(feature_spec.keys())

  def process(self, elem):
    if not isinstance(elem, dict):
      raise ValueError(
          'Element must be a dict(str, value). '
          'Given: {} {}'.format(elem, type(elem)))
    elem_features = set(elem.keys())
    if not self.feature_names.issubset(elem_features):
      raise ValueError(
          "Element features are missing from feature_spec keys. "
          'Given: {}; Features: {}'.format(
              list(elem_features), list(self.feature_names)))
    yield elem


def run(
    input_feature_spec,
    labels,
    feature_extraction,
    feature_scaling=None,
    eval_percent=20.0,
    beam_options=None,
    work_dir=None):
  """Runs the whole preprocessing step.

  This runs the feature extraction PTransform, validates that the data conforms
  to the schema provided, normalizes the features, and splits the dataset into
  a training and evaluation dataset.
  """

  # Populate optional arguments
  if not feature_scaling:
    feature_scaling = lambda inputs: inputs

  # Type checking
  if not isinstance(labels, list):
    raise ValueError(
        '`labels` must be list(str). '
        'Given: {} {}'.format(labels, type(labels)))

  if not isinstance(feature_extraction, beam.PTransform):
    raise ValueError(
        '`feature_extraction` must be {}. '
        'Given: {} {}'.format(beam.PTransform,
            feature_extraction, type(feature_extraction)))

  if not callable(feature_scaling):
    raise ValueError(
        '`feature_scaling` must be callable. '
        'Given: {} {}'.format(feature_scaling,
            type(feature_scaling)))

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

  # [START dataflow_molecules_create_pipeline]
  # Build and run a Beam Pipeline
  with beam.Pipeline(options=beam_options) as p, \
       beam_impl.Context(temp_dir=tft_temp_dir):
    # [END dataflow_molecules_create_pipeline]

    # [START dataflow_molecules_feature_extraction]
    # Transform and validate the input data matches the input schema
    dataset = (
        p
        | 'Feature extraction' >> feature_extraction
        # [END dataflow_molecules_feature_extraction]
        # [START dataflow_molecules_validate_inputs]
        | 'Validate inputs' >> beam.ParDo(ValidateInputData(
            input_feature_spec)))
    # [END dataflow_molecules_validate_inputs]

    # [START dataflow_molecules_analyze_and_transform_dataset]
    # Apply the tf.Transform preprocessing_fn
    input_metadata = dataset_metadata.DatasetMetadata(
        dataset_schema.from_feature_spec(input_feature_spec))

    dataset_and_metadata, transform_fn = (
        (dataset, input_metadata)
        | 'Feature scaling' >> beam_impl.AnalyzeAndTransformDataset(
            feature_scaling))
    dataset, metadata = dataset_and_metadata
    # [END dataflow_molecules_analyze_and_transform_dataset]

    # [START dataflow_molecules_split_to_train_and_eval_datasets]
    # Split the dataset into a training set and an evaluation set
    assert 0 < eval_percent < 100, 'eval_percent must in the range (0-100)'
    train_dataset, eval_dataset = (
        dataset
        | 'Split dataset' >> beam.Partition(
            lambda elem, _: int(random.uniform(0, 100) < eval_percent), 2))
    # [END dataflow_molecules_split_to_train_and_eval_datasets]

    # [START dataflow_molecules_write_tfrecords]
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
    # [END dataflow_molecules_write_tfrecords]

  return PreprocessData(
      input_feature_spec,
      labels,
      train_dataset_prefix + '*',
      eval_dataset_prefix + '*')


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--work-dir',
      required=True,
      help='Directory for staging and working files. '
           'This can be a Google Cloud Storage path.')
  args, pipeline_args = parser.parse_known_args()

  data_files_pattern = os.path.join(args.work_dir, 'data', '*.sdf')
  beam_options = PipelineOptions(pipeline_args, save_main_session=True)
  preprocess_data = run(
      pubchem.FEATURE_SPEC,
      pubchem.LABELS,
      # [START dataflow_molecules_feature_extraction_transform]
      pubchem.SimpleFeatureExtraction(pubchem.ParseSDF(data_files_pattern)),
      # [END dataflow_molecules_feature_extraction_transform]
      feature_scaling=pubchem.normalize_inputs,
      beam_options=beam_options,
      work_dir=args.work_dir)

  dump(preprocess_data, os.path.join(args.work_dir, 'PreprocessData'))
