# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Defines the Beam pipeline to preprocess the data."""

import csv

import apache_beam as beam
import tensorflow as tf


class CsvFileSource(beam.io.filebasedsource.FileBasedSource):

  def read_records(self, file_name, unused_range_tracker):
    self._file = tf.gfile.GFile(file_name)
    reader = csv.reader(self._file)
    for rec in reader:
      yield rec


def run(p, input_path):

  raw_data = (p
              | 'ReadTrainData' >> beam.io.Read(CsvFileSource(input_path)))
