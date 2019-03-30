# Copyright 2019 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Data ingest operations."""

# TODO(cezequiel): Remove when done using sns dataset
import seaborn as sns

from sklearn import model_selection

from trainer import metadata


def _feature_label_split(data, label_column):
  return data.loc[:, data.columns != label_column], data[label_column]


def read_from_bigquery(table_name):
  # TODO(cezequiel): Implement BQ reader
  data = sns.load_dataset('titanic')

  _ = table_name
  label_column = metadata.LABEL

  train, val = model_selection.train_test_split(data)
  x_train, y_train = _feature_label_split(train, label_column)
  x_val, y_val= _feature_label_split(val, label_column)

  return x_train, y_train, x_val, y_val
