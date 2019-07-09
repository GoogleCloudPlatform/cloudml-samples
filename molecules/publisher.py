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

# This is a sample publisher for the streaming predictions service.

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys

import pubchem

import apache_beam as beam

from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--topic',
      default='molecules-inputs',
      help='PubSub topic to publish molecules.')

  parser.add_argument(
      '--inputs-dir',
      required=True,
      help='Input directory where SDF data files are read from. '
           'This can be a Google Cloud Storage path.')

  args, pipeline_args = parser.parse_known_args()

  beam_options = PipelineOptions(
      pipeline_args,
      save_main_session=True,
      streaming=True,
  )

  project = beam_options.view_as(GoogleCloudOptions).project
  if not project:
    parser.print_usage()
    print('error: argument --project is required')
    sys.exit(1)

  data_files_pattern = os.path.join(args.inputs_dir, '*.sdf')
  topic_path = 'projects/{}/topics/{}'.format(project, args.topic)
  with beam.Pipeline(options=beam_options) as p:
    _ = (p
        | 'Read SDF files' >> pubchem.ParseSDF(data_files_pattern)
        | 'Print element' >> beam.Map(lambda elem: print(str(elem)[:70] + '...') or elem)
        | 'Publish molecules' >> beam.io.WriteToPubSub(topic=topic_path))
