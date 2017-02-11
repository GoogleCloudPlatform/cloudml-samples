# Copyright 2017 Google Inc. All Rights Reserved.
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
r"""A script for predicting using an Criteo model locally.

  # Using a model from the local filesystem:
  python local_predict.py --model_dir=output/${JOB_NAME}/model \
    data_predict.tsv

  # Using a model from GCS:
  python local_predict.py --model_dir=gs://${PROJECT_ID}-ml/criteo/model \
    data_predict.tsv
"""

import argparse
import collections
import json
import os
import sys

from google.cloud.ml import features
from google.cloud.ml import session_bundle


def local_predict(input_data, model_dir, metadata_file_name):
  """Runs prediction locally.

  Args:
    input_data: list of input files to run prediction on.
    model_dir: path to Tensorflow model folder.
    metadata_file_name: one of metadata.{json|yaml}.
  """

  session, _ = session_bundle.load_session_bundle_from_path(model_dir)
  # get the mappings between aliases and tensor names
  # for both inputs and outputs
  input_alias_map = json.loads(session.graph.get_collection('inputs')[0])
  output_alias_map = json.loads(session.graph.get_collection('outputs')[0])
  aliases, tensor_names = zip(*output_alias_map.items())

  metadata_path = os.path.join(model_dir, metadata_file_name)
  transformer = features.FeatureProducer(metadata_path)
  for input_file in input_data:
    with open(input_file) as f:
      feed_dict = collections.defaultdict(list)
      for line in f:
        preprocessed = transformer.preprocess(line)
        feed_dict[input_alias_map.values()[0]].append(
            preprocessed.SerializeToString())
      result = session.run(fetches=tensor_names, feed_dict=feed_dict)
      for row in zip(*result):
        print json.dumps({
            name: (value.tolist() if getattr(value, 'tolist', None) else
                   value)
            for name, value in zip(aliases, row)
        })


def parse_args(args):
  """Parses arguments specified on the command-line."""

  argparser = argparse.ArgumentParser('Predict on the Iris model.')

  argparser.add_argument(
      'input_data',
      nargs='+',
      help=('The input data file. Multiple files can be specified if more than '
            'one file is needed.'))
  argparser.add_argument(
      '--model_dir',
      dest='model_dir',
      help=('The path to the model where the tensorflow meta graph '
            'proto and checkpoint files are saved.'))
  argparser.add_argument(
      '--metadata_file_name',
      default='metadata.json',
      help='Name for the metadata file, one of metadata.{json|yaml}.')

  return argparser.parse_args(args)

if __name__ == '__main__':
  parsed_args = parse_args(sys.argv[1:])
  local_predict(**vars(parsed_args))
