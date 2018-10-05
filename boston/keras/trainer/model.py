# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import utils

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import models

tf.logging.set_verbosity(tf.logging.INFO)


def keras_estimator(model_dir, config, learning_rate, num_features):
  """Creates a Keras Sequential model with layers.

  Mean Squared Error (MSE) is a common loss function used for regression.
  A common regression metric is Mean Absolute Error (MAE).

  Args:
    model_dir: (str) file path where training files will be written.
    config: (tf.estimator.RunConfig) Configuration options to save model.
    learning_rate: (int) Learning rate.
    num_features: (int) Number of features.

  Returns:
    A keras.Model
  """
  model = models.Sequential()
  model.add(Dense(64, activation=tf.nn.relu, input_shape=(num_features,)))
  model.add(Dense(64, activation=tf.nn.relu))
  model.add(Dense(1))

  # Compile model with learning parameters.
  optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model, model_dir=model_dir, config=config)
  return estimator


def input_fn(x, y, batch_size, mode):
  """Input function for training and serving.

  Args:
    x: (numpy.array) Training or eval data.
    y: (numpy.array) Labels for training or eval data.
    batch_size: (int)
    mode: tf.estimator.ModeKeys mode

  Returns:
    A tf.estimator.
  """
  # Default settings for training.
  num_epochs = None
  shuffle = True

  # Override if this is eval.
  if mode == tf.estimator.ModeKeys.EVAL:
    num_epochs = 1
    shuffle = False

  y = np.asarray(y).astype('float32').reshape((-1, 1))
  return tf.estimator.inputs.numpy_input_fn(
      x,
      y=y,
      batch_size=batch_size,
      num_epochs=num_epochs,
      shuffle=shuffle,
      queue_capacity=50000)


def serving_input_fn():
  """Defines the features to be passed to the model during inference.

  Returns:
    A tf.estimator.export.ServingInputReceiver
  """
  feature_placeholder = tf.placeholder(tf.float32, [None, 13])
  features = feature_placeholder
  return tf.estimator.export.TensorServingInputReceiver(features,
                                                        feature_placeholder)


def train_and_evaluate(output_dir, hparams):
  """Helper function: Trains and evaluate model.

  Args:
    output_dir: (str) File path where training files will be written.
    hparams: (dict) Command line parameters passed from task.py
  """
  # Load data.
  (train_data,
   train_labels), (test_data,
                   test_labels) = utils.load_data(path=hparams['dataset_file'])

  # Shuffle data.
  order = np.argsort(np.random.random(train_labels.shape))
  train_data = train_data[order]
  train_labels = train_labels[order]

  # Normalize data.
  train_data, test_data = utils.normalize_data(train_data, test_data)

  # Create estimator.
  run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
  estimator = keras_estimator(
      model_dir=output_dir,
      config=run_config,
      learning_rate=hparams['learning_rate'],
      num_features=train_data.shape[1])
  train_steps = hparams['num_epochs'] * len(train_data) / hparams['batch_size']
  train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn(
          train_data,
          train_labels,
          hparams['batch_size'],
          mode=tf.estimator.ModeKeys.TRAIN),
      max_steps=train_steps)
  # Create EvalSpec.
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=input_fn(
          test_data,
          test_labels,
          hparams['batch_size'],
          mode=tf.estimator.ModeKeys.EVAL),
      steps=None,
      exporters=exporter,
      start_delay_secs=10,
      throttle_secs=10)

  # Start training.
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)