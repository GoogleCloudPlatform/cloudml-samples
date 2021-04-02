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

import argparse

import numpy as np
from . import model
from . import utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def get_args():
  """Argument parser.

	Returns:
	  Dictionary of arguments.
	"""
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--job-dir',
    type=str,
    required=True,
    help='GCS location to write checkpoints and export models')
  parser.add_argument(
    '--train-file',
    type=str,
    required=True,
    help='Training file local or GCS')
  parser.add_argument(
    '--train-labels-file',
    type=str,
    required=True,
    help='Training labels file local or GCS')
  parser.add_argument(
    '--test-file',
    type=str,
    required=True,
    help='Test file local or GCS')
  parser.add_argument(
    '--test-labels-file',
    type=str,
    required=True,
    help='Test file local or GCS')
  parser.add_argument(
    '--num-epochs',
    type=float,
    default=5,
    help='number of times to go through the data, default=5')
  parser.add_argument(
    '--batch-size',
    default=128,
    type=int,
    help='number of records to read during each training step, default=128')
  parser.add_argument(
    '--learning-rate',
    default=.01,
    type=float,
    help='learning rate for gradient descent, default=.001')
  parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')
  return parser.parse_args()


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

  # Define training steps.
  train_steps = hparams.num_epochs * len(
      train_images) / hparams.batch_size
  # Create TrainSpec.
  train_labels = np.asarray(train_labels).astype('int').reshape((-1, 1))
  train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: model.input_fn(
          train_images,
          train_labels,
          hparams.batch_size,
          mode=tf.estimator.ModeKeys.TRAIN),
      max_steps=train_steps)

  # Create EvalSpec.
  exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
  # Shape numpy array.
  test_labels = np.asarray(test_labels).astype('int').reshape((-1, 1))
  eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: model.input_fn(
          test_images,
          test_labels,
          hparams.batch_size,
          mode=tf.estimator.ModeKeys.EVAL),
      steps=None,
      exporters=exporter,
      start_delay_secs=10,
      throttle_secs=10)

  # Define running config.
  run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
  # Create estimator.
  estimator = model.keras_estimator(
    model_dir=hparams.job_dir,
    config=run_config,
    learning_rate=hparams.learning_rate)
  # Start training
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':

  args = get_args()
  tf.logging.set_verbosity(args.verbosity)

  hparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(hparams)