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


from sklearn import compose
from sklearn import ensemble
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing

import numpy as np

from trainer import metadata


def get_estimator(flags):
  # TODO: Allow pre-processing to be configurable through flags
  classifier = ensemble.RandomForestClassifier(**flags)

  numeric_transformer = pipeline.Pipeline([
      ('imputer', impute.SimpleImputer(strategy='median')),
      ('scaler', preprocessing.StandardScaler()),
  ])

  numeric_log_transformer = pipeline.Pipeline([
      ('imputer', impute.SimpleImputer(strategy='median')),
      ('log', preprocessing.FunctionTransformer(
          func=np.log1p, inverse_func=np.expm1, validate=True)),
      ('scaler', preprocessing.StandardScaler()),
  ])

  numeric_bin_transformer = pipeline.Pipeline([
      ('imputer', impute.SimpleImputer(strategy='median')),
      ('bin', preprocessing.KBinsDiscretizer(n_bins=5, encode='onehot-dense')),
  ])

  categorical_transformer = pipeline.Pipeline([
      ('imputer', impute.SimpleImputer(
          strategy='constant', fill_value='missing')),
      ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)),
  ])

  preprocessor = compose.ColumnTransformer([
      ('numeric', numeric_transformer, metadata.NUMERIC_FEATURES),
      ('numeric', numeric_log_transformer, metadata.NUMERIC_FEATURES),
      ('numeric', numeric_bin_transformer, metadata.NUMERIC_FEATURES),
      ('categorical', categorical_transformer, metadata.CATEGORICAL_FEATURES),
  ])

  estimator = pipeline.Pipeline([
      ('preprocessor', preprocessor),
      ('classifier', classifier),
  ])

  return estimator
