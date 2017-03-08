# Copyright 2016 Google Inc. All Rights Reserved.
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
r"""A script for predicting using an MNIST model locally.

  # Using a model from the local filesystem:
  python local_predict.py --model_dir=output/${JOB_NAME}/model \
    data/eval_sample.tfrecord

  # Using a model from GCS:
  python local_predict.py \
    --model_dir=gs://${PROJECT_ID}-ml/mnist/${JOB_NAME}/model \
    data/eval_sample.tfrecord
"""

import argparse
import collections
import json
import os
import subprocess

from tensorflow.python.client import session
from tensorflow.python.lib.io import tf_record
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants


def _default_project():
  get_project = ['gcloud', 'config', 'list', 'project',
                 '--format=value(core.project)']

  with open(os.devnull, 'w') as dev_null:
    return subprocess.check_output(get_project, stderr=dev_null).strip()


def local_predict(args):
  """Runs prediction locally."""

  sess = session.Session()
  _ = loader.load(sess, [tag_constants.SERVING], args.model_dir)

  # get the mappings between aliases and tensor names
  # for both inputs and outputs
  input_alias_map = json.loads(sess.graph.get_collection('inputs')[0])
  output_alias_map = json.loads(sess.graph.get_collection('outputs')[0])
  aliases, tensor_names = zip(*output_alias_map.items())

  for input_file in args.input:
    feed_dict = collections.defaultdict(list)
    for line in tf_record.tf_record_iterator(input_file):
      feed_dict[input_alias_map['examples_bytes']].append(line)

    if args.dry_run:
      print 'Feed data dict %s to graph and fetch %s' % (
          feed_dict, tensor_names)
    else:
      result = sess.run(fetches=tensor_names, feed_dict=feed_dict)
      for row in zip(*result):
        print json.dumps(
            {name: (value.tolist() if getattr(value, 'tolist', None) else value)
             for name, value in zip(aliases, row)})


def parse_args():
  """Parses arguments specified on the command-line."""

  argparser = argparse.ArgumentParser('Predict on the MNIST model.')

  argparser.add_argument(
      'input',
      nargs='+',
      help=('The input data file/file patterns. Multiple '
            'files can be specified if more than one file patterns is needed.'))

  argparser.add_argument(
      '--model_dir',
      dest='model_dir',
      help=('The path to the model where the tensorflow meta graph '
            'proto and checkpoint files are saved.'))

  argparser.add_argument(
      '--dry_run',
      action='store_true',
      help='Instead of executing commands, prints them.')

  return argparser.parse_args()


if __name__ == '__main__':
  arguments = parse_args()
  local_predict(arguments)
