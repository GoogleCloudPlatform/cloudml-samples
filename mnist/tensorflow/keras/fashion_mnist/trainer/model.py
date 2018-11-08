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
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras import models

tf.logging.set_verbosity(tf.logging.INFO)


def keras_estimator(model_dir, config, learning_rate):
  """Creates a Keras Sequential model with layers.

  Args:
    model_dir: (str) file path where training files will be written.
    config: (tf.estimator.RunConfig) Configuration options to save model.
    learning_rate: (int) Learning rate.

  Returns:
    A keras.Model
  """
  model = models.Sequential()
  model.add(Flatten(input_shape=(28, 28)))
  model.add(Dense(128, activation=tf.nn.relu))
  model.add(Dense(10, activation=tf.nn.softmax))

  # Compile model with learning parameters.
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  model.compile(
      optimizer=optimizer,
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model, model_dir=model_dir, config=config)
  return estimator


def input_fn(features, labels, batch_size, mode):
  """Input function.

  Args:
    features: (numpy.array) Training or eval data.
    labels: (numpy.array) Labels for training or eval data.
    batch_size: (int)
    mode: tf.estimator.ModeKeys mode

  Returns:
    A tf.estimator.
  """
  # Default settings for training.
  if labels is None:
    inputs = features
  else:
    # Change numpy array shape.
    labels = np.asarray(labels).astype('int').reshape((-1, 1))
    inputs = (features, labels)
  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices(inputs)
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
    dataset = dataset.batch(batch_size)
  return dataset.make_one_shot_iterator().get_next()


def serving_input_fn():
  """Defines the features to be passed to the model during inference.

  Expects already tokenized and padded representation of sentences

  Returns:
    A tf.estimator.export.ServingInputReceiver
  """
  feature_placeholder = tf.placeholder(tf.float32, [None, 784])
  features = feature_placeholder
  return tf.estimator.export.TensorServingInputReceiver(features,
                                                        feature_placeholder)


def train_and_evaluate(hparams):
  """Helper function: Trains and evaluates model.

  Args:
    hparams: (dict) Command line parameters passed from task.py
  """
  # Loads data.
  (train_images, train_labels), (test_images, test_labels) = \
      utils.prepare_data(train_file=hparams.train_file,
                         train_labels_file=hparams.train_labels_file,
                         test_file=hparams.test_file,
                         test_labels_file=hparams.test_labels_file)

  # Scale values to a range of 0 to 1.
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  # Create estimator.
  run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
  estimator = keras_estimator(
      model_dir=hparams.output_dir,
      config=run_config,
      learning_rate=hparams.learning_rate)
  train_steps = hparams.num_epochs * len(
      train_images) / hparams.batch_size
  # Create TrainSpec.
  train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: input_fn(
          train_images,
          train_labels,
          hparams.batch_size,
          mode=tf.estimator.ModeKeys.TRAIN),
      max_steps=train_steps)

  # Create EvalSpec.
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: input_fn(
          test_images,
          test_labels,
          hparams.batch_size,
          mode=tf.estimator.ModeKeys.EVAL),
      steps=None,
      exporters=exporter,
      start_delay_secs=10,
      throttle_secs=10)

  # Start training
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
