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
import os

import model

import tensorflow as tf


def train_and_evaluate(args):
  """Run the training and evaluate using the high level API"""
  train_input = model._make_training_input_fn(
      args.tft_working_dir,
      args.train_filebase,
      num_epochs=args.num_epochs,
      batch_size=args.train_batch_size,
      buffer_size=args.train_buffer_size,
      prefetch_buffer_size=args.train_prefetch_buffer_size)

  # Don't shuffle evaluation data.
  eval_input = model._make_training_input_fn(
      args.tft_working_dir,
      args.eval_filebase,
      shuffle=False,
      batch_size=args.eval_batch_size,
      buffer_size=1,
      prefetch_buffer_size=args.eval_prefetch_buffer_size)

  train_spec = tf.estimator.TrainSpec(
      train_input, max_steps=args.train_steps)

  exporter = tf.estimator.FinalExporter(
      'tft_classifier', model._make_serving_input_fn(args.tft_working_dir))

  eval_spec = tf.estimator.EvalSpec(
      eval_input,
      steps=args.eval_steps,
      exporters=[exporter],
      name='tft_classifier-eval')

  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=args.job_dir)

  print('model dir {}'.format(run_config.model_dir))
  estimator = model.build_estimator(
      config=run_config,
      tft_working_dir=args.tft_working_dir,
      embedding_size=args.embedding_size,
      # Construct layers sizes with exponential decay.
      hidden_units=[
          max(2, int(args.first_layer_size * args.scale_factor**i))
          for i in range(args.num_layers)
      ],
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--tft-working-dir',
      help='GCS or local paths to directory pointed by tf transform pipeline',
      required=True)
  parser.add_argument(
      '--train-filebase',
      help='Path to training data as in preprocessing.py',
      required=True)
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=256)
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True)
  parser.add_argument(
      '--train-buffer-size',
      help='Buffer size for the shuffle',
      type=int,
      default=None)
  parser.add_argument(
      '--train-prefetch-buffer-size',
      help='Number of example to prefetch',
      type=int,
      default=1)
  parser.add_argument(
      '--eval-filebase',
      help='Path to eval data as in preprocessing.py',
      required=True)
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for training steps',
      type=int,
      default=256)
  parser.add_argument(
      '--eval-prefetch-buffer-size',
      help='Number of example to prefetch',
      type=int,
      default=1)
  # Training arguments
  parser.add_argument(
      '--embedding-size',
      help='Number of embedding dimensions for categorical columns',
      default=8,
      type=int)
  parser.add_argument(
      '--first-layer-size',
      help='Number of nodes in the first layer of the DNN',
      default=100,
      type=int)
  parser.add_argument(
      '--num-layers',
      help='Number of layers in the DNN',
      default=4,
      type=int)
  parser.add_argument(
      '--scale-factor',
      help='How quickly should the size of the layers in the DNN decay',
      default=0.7,
      type=float)
  parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.""",
      type=int)
  parser.add_argument(
      '--train-steps',
      help='Steps to run the training job for. Use for distributed training',
      type=int)
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int)
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO',
      help='Set logging verbosity')

  args = parser.parse_args()

  # Set python level verbosity.
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  train_and_evaluate(args)
