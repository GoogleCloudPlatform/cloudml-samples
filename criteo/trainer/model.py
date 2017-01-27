# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Feature spec for the Criteo model."""

import google.cloud.ml.features as features

COUNT_RANGE = range(1, 14)
CATEGORY_RANGE = range(14, 40)


def criteo_features(frequency_threshold):
  """Creates a dictionary-style feature-spec for Criteo."""

  csv_headers = ['clicked']
  feature_dict = {'clicked': features.target('clicked').discrete()}

  for column_idx in COUNT_RANGE:
    name = 'int-feature-{}'.format(column_idx)
    csv_headers.append(name)
    feature_dict[name] = features.numeric(name, default=-1).identity('int64')
  for column_idx in CATEGORY_RANGE:
    name = 'categorical-feature-{}'.format(column_idx)
    csv_headers.append(name)
    column = features.categorical(name, frequency_threshold=frequency_threshold)
    feature_dict[name] = column
  return feature_dict, csv_headers
