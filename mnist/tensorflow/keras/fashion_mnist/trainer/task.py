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
import logging
import os
import sys

from . import model

from tensorflow.contrib.training.python.training import hparam


def get_args():
  """Argument parser.

	Returns:
	  Dictionary of arguments.
	"""
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--output_dir',
    type=str,
    required=True,
    help='GCS location to write checkpoints and export models')
  parser.add_argument(
    '--train_file',
    type=str,
    required=True,
    help='Training file local or GCS')
  parser.add_argument(
    '--train_labels_file',
    type=str,
    required=True,
    help='Training labels file local or GCS')
  parser.add_argument(
    '--test_file',
    type=str,
    required=True,
    help='Test file local or GCS')
  parser.add_argument(
    '--test_labels_file',
    type=str,
    required=True,
    help='Test file local or GCS')
  parser.add_argument(
    '--num_epochs',
    type=float,
    default=5,
    help='number of times to go through the data, default=5')
  parser.add_argument(
    '--batch_size',
    default=128,
    type=int,
    help='number of records to read during each training step, default=128')
  parser.add_argument(
    '--learning_rate',
    default=.01,
    type=float,
    help='learning rate for gradient descent, default=.001')
  return parser.parse_args()


def _setup_logging():
  """Sets up logging."""
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  # Set tf logging to avoid duplicate logging. If the handlers are not removed,
  # then we will have duplicate logging:
  # From tf loggging written to stderr stream, and
  # From python logger written to stdout stream.
  tf_logger = logging.getLogger('tensorflow')
  while tf_logger.handlers:
    tf_logger.removeHandler(tf_logger.handlers[0])
   # Redirect INFO logs to stdout
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.INFO)
  logger.addHandler(stdout_handler)
   # Suppress C++ level warnings.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
  args = get_args()
  _setup_logging()

  hparams = hparam.HParams(**args.__dict__)
  model.train_and_evaluate(hparams)