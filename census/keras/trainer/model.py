# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Implements the Keras Sequential model."""

import keras
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.utils import np_utils
from keras.backend import relu, softmax

#Python2/3 compatibility imports
from six.moves.urllib import parse as urlparse
from builtins import range

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

# csv columns in the input file
CSV_COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race',
               'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'income_bracket')

CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''],
                       [''], [0], [0], [0], [''], ['']]

# Categorical columns with vocab size
# native_country and fnlwgt are ignored
CATEGORICAL_COLS = (('education', 16), ('marital_status', 7),
                    ('relationship', 6), ('workclass', 9), ('occupation', 15),
                    ('gender', [' Male', ' Female']), ('race', 5))

CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss',
                   'hours_per_week')

LABELS = [' <=50K', ' >50K']
LABEL_COLUMN = 'income_bracket'

UNUSED_COLUMNS = set(CSV_COLUMNS) - set(
    list(zip(*CATEGORICAL_COLS))[0] + CONTINUOUS_COLS + (LABEL_COLUMN,))


def model_fn(input_dim,
             labels_dim,
             hidden_units=[100, 70, 50, 20],
             learning_rate=0.001):
  """Create a Keras Sequential model with layers."""
  model = models.Sequential()

  for units in hidden_units:
    model.add(layers.Dense(units=units,
                           input_dim=input_dim,
                           activation=relu))
    input_dim = units

  # Add a dense final layer with sigmoid function
  model.add(layers.Dense(labels_dim, activation=softmax))
  compile_model(model, learning_rate)
  return model

def compile_model(model, learning_rate):
  model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                metrics=['accuracy'])
  return model

def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                    outputs={'income': model.outputs[0]})

  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()

def to_numeric_features(features,feature_cols=None):
  """Convert the pandas input features to numeric values.
     Args:
        features: Input features in the data
          age (continuous)
          workclass (categorical)
          fnlwgt (continuous)
          education (categorical)
          education_num (continuous)
          marital_status (categorical)
          occupation (categorical)
          relationship (categorical)
          race (categorical)
          gender (categorical)
          capital_gain (continuous)
          capital_loss (continuous)
          hours_per_week (continuous)
          native_country (categorical)

        feature_cols: Column list of converted features to be returned.
            Optional, may be used to ensure schema consistency over multiple executions.


  """

  for col in CATEGORICAL_COLS:
    features = pd.concat([features, pd.get_dummies(features[col[0]], drop_first = True)], axis = 1)
    features.drop(col[0], axis = 1, inplace = True)

  # Remove the unused columns from the dataframe
  for col in UNUSED_COLUMNS:
    features.pop(col)

  #Re-index dataframe (in case categories list changed from the previous dataset)
  if feature_cols is not None:
      features = features.T.reindex(feature_cols).T.fillna(0)

  return features

def generator_input(input_file, chunk_size, batch_size=64):
  """Generator function to produce features and labels
     needed by keras fit_generator.
  """

  feature_cols=None
  while True:
      input_reader = pd.read_csv(tf.gfile.Open(input_file[0]),
                               names=CSV_COLUMNS,
                               chunksize=chunk_size,
                               na_values=" ?")

      for input_data in input_reader:
        input_data = input_data.dropna()
        label = pd.get_dummies(input_data.pop(LABEL_COLUMN))

        input_data = to_numeric_features(input_data,feature_cols)

        #Retains schema for next chunk processing
        if feature_cols is None:
            feature_cols=input_data.columns

        idx_len=input_data.shape[0]
        for index in range(0,idx_len,batch_size):
            yield (input_data.iloc[index:min(idx_len,index+batch_size)], label.iloc[index:min(idx_len,index+batch_size)])
