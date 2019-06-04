# Copyright 2019 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This file contains the shared functionality for preprocess.py and predict.py

from __future__ import absolute_import

import json
import logging
import pprint

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft

from apache_beam.io import filebasedsource
from tensorflow_transform.tf_metadata import dataset_schema

from pubchem import sdf


FEATURE_SPEC = {
    # Features (inputs)
    'TotalC': tf.io.FixedLenFeature([], tf.int64),
    'TotalH': tf.io.FixedLenFeature([], tf.int64),
    'TotalO': tf.io.FixedLenFeature([], tf.int64),
    'TotalN': tf.io.FixedLenFeature([], tf.int64),

    # Labels (outputs/predictions)
    'Energy': tf.io.FixedLenFeature([], tf.float32),
}

LABELS = ['Energy']


class ParseSDF(beam.PTransform):
  def __init__(self, file_patterns):
    super(ParseSDF, self).__init__()
    if isinstance(file_patterns, str):
      file_patterns = [file_patterns]
    self.file_patterns = file_patterns

  def expand(self, pcollection):
    def parse_molecules(filename):
      with tf.gfile.Open(filename) as f:
        for json_molecule in sdf.parse_molecules(f):
          yield json_molecule

    return (
        pcollection
        | 'Create file patterns' >> beam.Create(self.file_patterns)
        | 'Expand file patterns' >> beam.FlatMap(tf.gfile.Glob)
        | 'Parse molecules' >> beam.ParDo(parse_molecules)
    )


class FormatMolecule(beam.DoFn):
  def process(self, json_molecule):
    """Helper function to format a molecule.

    For more information, please check: http://c4.cabrillo.edu/404/ctfile.pdf
    """
    try:
      # The molecules are currently encoded in JSON so they can be read from
      # files directly as well as through PubSub messages for streaming
      # predictions. PubSub source expects everything as a single string.
      # To lower the network traffic, the JSON strings could be compressed with
      # the zlib library and encoded to base64 on the ParseSDF source, and
      # decoded/decompressed here. However, they would no longer be human
      # readable.
      molecule = json.loads(json_molecule)
      counts_line = molecule[sdf.MDF_SECTION][0]
      total_atoms = int(counts_line[0:3])
      total_bonds = int(counts_line[3:6])
      is_chiral = bool(counts_line[12:15])
      ctab_version = counts_line[33:39].strip()

      atoms = []
      start = 1
      end = start + total_atoms
      for line in molecule[sdf.MDF_SECTION][start:end]:
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
      for line in molecule[sdf.MDF_SECTION][start:end]:
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
      molecule.pop(sdf.MDF_SECTION)
      yield molecule

    except Exception:
      logging.exception(pprint.pformat(molecule))


class CountAtoms(beam.DoFn):
  def count_by_atom_symbol(self, molecule, atom_symbol):
    # For every atom in the molecule, check if the atom type equals the
    # element we're trying to count. If True, it will be casted to a 1,
    # otherwise it will be a 0, then sum all of them.
    return sum(int(atom['atom_symbol'] == atom_symbol)
        for atom in molecule['atoms'])

  def process(self, molecule):
    try:
      uid = int(molecule['<PUBCHEM_COMPOUND_CID>'][0])
      label = float(molecule['<PUBCHEM_MMFF94_ENERGY>'][0])
      result = {
        'ID': uid,
        'TotalC': self.count_by_atom_symbol(molecule, 'C'),
        'TotalH': self.count_by_atom_symbol(molecule, 'H'),
        'TotalO': self.count_by_atom_symbol(molecule, 'O'),
        'TotalN': self.count_by_atom_symbol(molecule, 'N'),
        'Energy': label,
      }
      yield result
    except Exception as e:
      logging.info('Invalid molecule, skipping: {}'.format(molecule), exc_info=True)


# [START dataflow_molecules_simple_feature_extraction]
class SimpleFeatureExtraction(beam.PTransform):
  """The feature extraction (element-wise transformations).

  We create a `PTransform` class. This `PTransform` is a bundle of
  transformations that can be applied to any other pipeline as a step.

  We'll extract all the raw features here. Due to the nature of `PTransform`s,
  we can only do element-wise transformations here. Anything that requires a
  full-pass of the data (such as feature scaling) has to be done with
  tf.Transform.
  """
  def __init__(self, source):
    super(SimpleFeatureExtraction, self).__init__()
    self.source = source

  def expand(self, p):
    # Return the preprocessing pipeline. In this case we're reading the PubChem
    # files, but the source could be any Apache Beam source.
    return (p
        | 'Read raw molecules' >> self.source
        | 'Format molecule' >> beam.ParDo(FormatMolecule())
        | 'Count atoms' >> beam.ParDo(CountAtoms())
    )
# [END dataflow_molecules_simple_feature_extraction]


# [START dataflow_molecules_normalize_inputs]
def normalize_inputs(inputs):
  """Preprocessing function for tf.Transform (full-pass transformations).

  Here we will do any preprocessing that requires a full-pass of the dataset.
  It takes as inputs the preprocessed data from the `PTransform` we specify, in
  this case `SimpleFeatureExtraction`.

  Common operations might be scaling values to 0-1, getting the minimum or
  maximum value of a certain field, creating a vocabulary for a string field.

  There are two main types of transformations supported by tf.Transform, for
  more information, check the following modules:
    - analyzers: tensorflow_transform.analyzers.py
    - mappers:   tensorflow_transform.mappers.py

  Any transformation done in tf.Transform will be embedded into the TensorFlow
  model itself.
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
# [END dataflow_molecules_normalize_inputs]
