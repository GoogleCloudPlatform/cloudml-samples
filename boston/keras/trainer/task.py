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

import argparse

from . import model

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help='GCS location to write checkpoints and export models')
  parser.add_argument(
      '--dataset-file',
      type=str,
      required=True,
      help='Dataset file local or GCS')
  parser.add_argument(
      '--test_split',
      type=float,
      default=0.2,
      help='Split between training and test, default=0.2')
  parser.add_argument(
      '--num_epochs',
      type=float,
      default=500,
      help='number of times to go through the data, default=500')
  parser.add_argument(
      '--batch_size',
      default=128,
      type=int,
      help='number of records to read during each training step, default=128')
  parser.add_argument(
      '--learning_rate',
      default=.001,
      type=float,
      help='learning rate for gradient descent, default=.001')

  parse_args, _ = parser.parse_known_args()
  hparams = parse_args.__dict__
  output_dir = hparams.pop('output_dir')

  model.train_and_evaluate(output_dir, hparams)