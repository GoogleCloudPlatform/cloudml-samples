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
import dill as pickle
import os
import random
import sys
import tempfile

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as beam_impl

from apache_beam.io import filebasedsource
from apache_beam.io import tfrecordio
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


DEFAULT_TEMP_DIR = os.path.join(
    tempfile.gettempdir(), 'cloudml-samples', 'molecules')

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


class PreprocessData(object):
  def __init__(
      self,
      labels,
      input_feature_spec,
      feature_spec,
      transform_fn_dir,
      train_files_pattern,
      eval_files_pattern):

    self.labels = labels
    self.input_feature_spec = input_feature_spec
    self.feature_spec = feature_spec
    self.transform_fn_dir = transform_fn_dir
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


class PubChemSource(filebasedsource.FileBasedSource):
  """This is a custom Source to parse through SDF files.

  It extends a FileBasedSource, but it won't use the TextSource implementation
  since every record in a PubChem file is multiple lines long.
  """
  MOLECULE_START = '-OEChem-'
  MOLECULE_END = '$$$$'
  MDF_SECTION = 'MDF'
  SECTION_PREFIX = '<PUBCHEM_'

  def parse_molecules(self, f):
    """Generator that yields raw molecules."""
    molecule = None
    section = None
    for raw_line in f:
      line = raw_line.lstrip('>').strip()
      if not line:
        continue

      # If we find a new molecule section, yield the last molecule we were
      # parsing and initialize a new one
      if self.MOLECULE_START in line:
        molecule = {}
        section = molecule[self.MDF_SECTION] = []
        continue

      if molecule is None:
        continue

      # This is the start of a new section
      if line.startswith(self.SECTION_PREFIX):
        section = molecule[line] = []

      # This delimits the end of a molecule
      elif line == self.MOLECULE_END:
        section = None
        yield self.format_molecule(molecule)
        molecule = None

      # It didn't match anything else, so it must be content for a section
      elif section is not None:
        section.append(raw_line)

    # If there's a last unprocessed molecule, yield it
    if molecule is not None:
      yield self.format_molecule(molecule)

  def format_molecule(self, molecule):
    """Helper function to format a molecule.

    For more information, please check: http://c4.cabrillo.edu/404/ctfile.pdf
    """
    try:
      counts_line = molecule[self.MDF_SECTION][0]
      total_atoms = int(counts_line[0:3])
      total_bonds = int(counts_line[3:6])
      is_chiral = bool(counts_line[12:15])
      ctab_version = counts_line[33:39].strip()

      atoms = []
      start = 1
      end = start + total_atoms
      for line in molecule[self.MDF_SECTION][start:end]:
        atoms.append({
          'x': float(line[ 0:10]),
          'y': float(line[10:20]),
          'z': float(line[20:30]),
          'atom_symbol': line[31:34].strip(),
          'mass_difference': int(line[34:36]),
          'charge': int(line[36:39]),
          'atom_stereo_parity': int(line[39:42]),
          'hydrogen_count': int(line[42:45]),
          'stereo_care_box': int(line[45:48]),
          'valence': int(line[48:51]),
          'h0_designator': int(line[51:54]),
          'atom_atom_mapping_number': int(line[60:63]),
          'inversion_retention': int(line[63:66]),
          'exact_change_flag': int(line[66:69]),
        })

      bonds = []
      start = end
      end = start + total_bonds
      for line in molecule[self.MDF_SECTION][start:end]:
        bonds.append({
          'first_atom_number': int(line[0:3]),
          'second_atom_number': int(line[3:6]),
          'bond_type': int(line[6:9]),
          'bond_stereo': int(line[9:12]),
          'bond_topology': int(line[15:18]),
          'reacting_center_status': int(line[18:21]),
        })

      molecule.update({
          'is_chiral': is_chiral,
          'ctab_version': ctab_version,
          'atoms': atoms,
          'bonds': bonds,
      })
      molecule.pop(self.MDF_SECTION)
      return molecule

    except Exception:
      import pprint
      s = pprint.pformat(sections)
      logging.exception('Error parsing molecule:\n{}'.format(s))

  def read_records(self, file_name, range_tracker):
    """This yields a dictionary with the sections of the file that we're
    interested in. The `range_tracker` allows us to mark the position where
    possibly another worker left, so we make sure to start from there.
    """
    with self.open_file(file_name) as f:
      f.seek(range_tracker.start_position() or 0)
      while range_tracker.try_claim(f.tell()):
        for elem in self.parse_molecules(f):
          yield elem


class PreprocessingPipeline(beam.PTransform):
  """The preprocessing pipeline (element-wise transformations).

  We create a `PTransform` class. This `PTransform` is a bundle of
  transformations that cann be applied to any other pipeline as a step.

  We'll do all the preprocessing needed here. Due to the nature of
  `PTransform`s, we can only do element-wise transformations here. Anything that
  requires a full-pass of the data has to be done with tf.Transform.
  """
  def __init__(self, data_files_pattern):
    super(PreprocessingPipeline, self).__init__()
    self.data_files_pattern = data_files_pattern

  def expand(self, p):
    # Helper functions
    def count_by_atom_symbol(elem, atom_symbol):
      # For every atom in the molecule, check if the atom type equals the
      # element we're trying to count. If True, it will be casted to a 1,
      # otherwise it will be a 0, then sum all of them.
      return sum(int(atom['atom_symbol'] == atom_symbol)
          for atom in elem['atoms'])

    def count_atoms(elem):
      label = float(elem['<PUBCHEM_MMFF94_ENERGY>'][0])
      return {
        'TotalC': count_by_atom_symbol(elem, 'C'),
        'TotalH': count_by_atom_symbol(elem, 'H'),
        'TotalO': count_by_atom_symbol(elem, 'O'),
        'TotalN': count_by_atom_symbol(elem, 'N'),
        'Energy': label,
      }

    # Return the preprocessing pipeline. In this case we're reading the PubChem
    # files, but the source could be any Apache Beam source.
    return (
        p
        | 'Read PubChem files' >> beam.io.Read(
            PubChemSource(self.data_files_pattern))
        | 'Count Atoms' >> beam.Map(count_atoms))


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
    if elem_keys != self.schema_keys:
      raise ValueError(
          "Element keys don't match the schema keys. "
          'Given: {}; Schema: {}'.format(
              list(elem_keys), list(self.schema_keys)))
    yield elem


def full_pass_preprocessing_fn(inputs):
  """Preprocessing function for tf.Transform (full-pass transformations).

  Here we will do any
  preprocessing that requires a full-pass of the dataset. It takes as inputs the
  preprocessed data from the `PTransform` we specify, in this case
  `PreprocessingPipeline`.

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


def run(
    input_schema,
    labels,
    preprocessing_ptransform,
    full_pass_preprocessing_fn=None,
    eval_percent=20.0,
    beam_options=None,
    temp_dir=None,
    tft_temp_dir=None,
    train_dataset_dir=None,
    eval_dataset_dir=None):
  """Runs the whole preprocessing step.

  This runs the preprocessing PTransform, validates that the data conforms to
  the schema provided, does the full-pass preprocessing step and generates the
  input functions needed to train and evaluate the TensorFlow model.
  """

  # Populate optional arguments
  if not full_pass_preprocessing_fn:
    full_pass_preprocessing_fn = lambda inputs: inputs

  if not temp_dir:
    temp_dir = tempfile.mkdtemp(prefix='tensorflow_model')

  if not tft_temp_dir:
    tft_temp_dir = os.path.join(temp_dir, 'tft_temp')

  if not train_dataset_dir:
    train_dataset_dir = os.path.join(temp_dir, 'train_dataset')

  if not eval_dataset_dir:
    eval_dataset_dir = os.path.join(temp_dir, 'eval_dataset')

  # Type checking
  if not isinstance(labels, list):
    raise ValueError(
        '`labels` must be list(str). '
        'Given: {} {}'.format(labels, type(labels)))

  if not isinstance(preprocessing_ptransform, beam.PTransform):
    raise ValueError(
        '`preprocessing_ptransform` must be {}. '
        'Given: {} {}'.format(beam.PTransform,
            preprocessing_ptransform, type(preprocessing_ptransform)))

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

  if tf.gfile.Exists(temp_dir):
    tf.gfile.DeleteRecursively(temp_dir)

  # Build and run a Beam Pipeline
  input_metadata = dataset_metadata.DatasetMetadata(
      dataset_schema.Schema(input_schema))

  with beam.Pipeline(options=beam_options) as p, \
       beam_impl.Context(temp_dir=tft_temp_dir):

    # Transform and validate the input data matches the input schema
    dataset = (
        p
        | 'Preprocessing' >> preprocessing_ptransform
        | 'ValidateInputData' >> beam.ParDo(ValidateInputData(input_metadata)))

    # Apply the tf.Transform preprocessing_fn
    dataset_and_metadata, transform_fn = (
        (dataset, input_metadata)
        | 'FullPassPreprocessing' >> beam_impl.AnalyzeAndTransformDataset(
            full_pass_preprocessing_fn))

    dataset, metadata = dataset_and_metadata

    # Split the dataset into a training set and an evaluation set
    assert 0 < eval_percent < 100, 'eval_percent must in the range (0-100)'
    train_dataset, eval_dataset = (
        dataset
        | 'SplitDataset' >> beam.Partition(
            lambda elem, _: int(random.uniform(0, 100) < eval_percent), 2))

    # Write the datasets as TFRecords
    coder = example_proto_coder.ExampleProtoCoder(metadata.schema)

    train_dataset_prefix = os.path.join(train_dataset_dir, 'part')
    _ = (
        train_dataset
        | 'WriteTrainDataset' >> tfrecordio.WriteToTFRecord(
            train_dataset_prefix, coder))

    eval_dataset_prefix = os.path.join(eval_dataset_dir, 'part')
    _ = (
        eval_dataset
        | 'WriteEvalDataset' >> tfrecordio.WriteToTFRecord(
            eval_dataset_prefix, coder))

    # Write the transform_fn
    _ = (
        transform_fn
        | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(temp_dir))

  return PreprocessData(
      labels,
      input_metadata.schema.as_feature_spec(),
      metadata.schema.as_feature_spec(),
      os.path.join(temp_dir, transform_fn_io.TRANSFORM_FN_DIR),
      train_dataset_prefix + '*',
      eval_dataset_prefix + '*')


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--data-dir',
                      type=str,
                      default=os.path.join(DEFAULT_TEMP_DIR, 'data'),
                      help='Directory to load the data files from. '
                           'This can be a Google Cloud Storage path.')
  parser.add_argument('--temp-dir',
                      type=str,
                      default=os.path.join(DEFAULT_TEMP_DIR, 'temp'),
                      help='Directory for temporary files and preprocessed '
                           'datasets to. '
                           'This can be a Google Cloud Storage path.')
  parser.add_argument('--preprocess-data',
                      type=str,
                      default=os.path.join(DEFAULT_TEMP_DIR, 'PreprocessData'),
                      help='Path to store the PreprocessData object to. '
                           'This can be a Google Cloud Storage path.')
  parser.add_argument('--eval-percent',
                      type=float,
                      default=20.0,
                      help='Percentage of the dataset to use for the '
                           'evaluation set (0-100)')
  args, pipeline_args = parser.parse_known_args()

  beam_options = PipelineOptions(pipeline_args)
  beam_options.view_as(SetupOptions).save_main_session = True

  data_files_pattern = os.path.join(args.data_dir, '*')

  preprocess_data = run(
      INPUT_SCHEMA,
      LABELS,
      PreprocessingPipeline(data_files_pattern),
      full_pass_preprocessing_fn,
      eval_percent=args.eval_percent,
      beam_options=beam_options,
      temp_dir=args.temp_dir)

  dump(preprocess_data, args.preprocess_data)
