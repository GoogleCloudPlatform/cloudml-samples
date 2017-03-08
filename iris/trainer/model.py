# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Iris Classification Model.

Iris problem was first introduced by Fisher's 1936 paper,
The Use of Multiple Measurements in Taxonomic Problems.

This is a classification model with 3 possible output classes and 4 numeric
input features.  One of the classes is linearly separable, but the other two
are not.

This sample creates a model to solve the problem using a small neural net
with a single hidden layer.
"""
import json
import os

import google.cloud.ml.features as features


def runs_on_cloud():
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  return env.get('task', None)


class IrisFeatures(object):

  csv_columns = ('key', 'species', 'sepal_length', 'sepal_width',
                 'petal_length', 'petal_width')

  key = features.key('key')
  species = features.target('species').discrete()
  measurements = [
      features.numeric('sepal_length'), features.numeric('sepal_width'),
      features.numeric('petal_length'), features.numeric('petal_width')
  ]
