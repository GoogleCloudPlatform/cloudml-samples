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
import os
import random

import apache_beam as beam

from constants import constants


class CsvFileSource(beam.io.filebasedsource.FileBasedSource):
    """Beam source for CSV files."""

    def __init__(self, file_pattern, column_names):
        self._column_names = column_names
        super(self.__class__, self).__init__(file_pattern)

    def read_records(self, file_name, unused_range_tracker):
        self._file = self.open_file(file_name)
        reader = csv.reader(self._file)
        for rec in reader:
            res = {key: val for key, val in zip(self._column_names, rec)}
            yield res


def split_data(examples, train_fraction):
    """Splits the data into train/eval.

    Args:
      examples: A PCollection.
      train_fraction: fraction of examples to keep in the train set (float).
    """

    def partition_fn(data, n_partition):
        random_value = random.random()
        if random_value < train_fraction:
            return 0
        return 1

    examples_split = (examples
                      | "SplitData" >> beam.Partition(partition_fn, 2))
    return examples_split


class ConvertDictToCSV(beam.DoFn):
    """Takes a dictionary and converts it to csv format."""

    def __init__(self, ordered_fieldnames, separator=","):
        self._ordered_fieldnames = ordered_fieldnames
        self._separator = separator

    def process(self, input_dict):
        value_list = []
        for field in self._ordered_fieldnames:
            value_list.append(input_dict[field])
        yield self._separator.join(value_list)


def run(p, input_path, output_directory, train_fraction=0.8):
    """Runs the pipeline."""

    raw_data = (p | "ReadTrainData" >> beam.io.Read(CsvFileSource(
            input_path, column_names=constants.CSV_COLUMNS)))
    train_data, eval_data = split_data(raw_data, train_fraction)

    (train_data | "PrepareCSV_train" >> beam.ParDo(
        ConvertDictToCSV(ordered_fieldnames=constants.CSV_COLUMNS))
     | "Write_train" >> beam.io.WriteToText(
            os.path.join(output_directory, "output_data", "train"),
            file_name_suffix=".csv"))
    (eval_data | "PrepareCSV_eval" >> beam.ParDo(
        ConvertDictToCSV(ordered_fieldnames=constants.CSV_COLUMNS))
     | "Write_eval" >> beam.io.WriteToText(
            os.path.join(output_directory, "output_data", "eval"),
            file_name_suffix=".csv"))
