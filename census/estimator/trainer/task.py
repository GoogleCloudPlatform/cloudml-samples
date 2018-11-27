# Copyright 2016 Google LLC
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

import argparse
import json
import os

import tensorflow as tf

import trainer.model as model


def _get_session_config_from_env_var():
  """Returns a tf.ConfigProto instance that has appropriate device_filters set.

  """

  tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

  if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
      'index' in tf_config['task']):
    # Master should only communicate with itself and ps
    if tf_config['task']['type'] == 'master':
      return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
    # Worker should only communicate with itself and ps
    elif tf_config['task']['type'] == 'worker':
      return tf.ConfigProto(device_filters=[
          '/job:ps',
          '/job:worker/task:%d' % tf_config['task']['index']
      ])
  return None


def train_and_evaluate(args):
  """Run the training and evaluate using the high level API."""

  train_input = lambda: model.input_fn(
    args.train_files,
      num_epochs=args.num_epochs,
      batch_size=args.train_batch_size
  )

  # Don't shuffle evaluation data
  eval_input = lambda: model.input_fn(
      args.eval_files,
      batch_size=args.eval_batch_size,
      shuffle=False
  )

  train_spec = tf.estimator.TrainSpec(
      train_input, max_steps=args.train_steps)

  exporter = tf.estimator.FinalExporter(
      'census', model.SERVING_FUNCTIONS[args.export_format])
  eval_spec = tf.estimator.EvalSpec(
      eval_input,
      steps=args.eval_steps,
      exporters=[exporter],
      name='census-eval')

  run_config = tf.estimator.RunConfig(
      session_config=_get_session_config_from_env_var())
  run_config = run_config.replace(model_dir=args.job_dir)
  print('Model dir %s' % run_config.model_dir)
  estimator = model.build_estimator(
      embedding_size=args.embedding_size,
      # Construct layers sizes with exponential decay
      hidden_units=[
          max(2, int(args.first_layer_size * args.scale_factor**i))
          for i in range(args.num_layers)
      ],
      config=run_config)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--train-files',
      help='GCS file or local paths to training data',
      nargs='+',
      default='gs://cloud-samples-data/ml-engine/census/data/adult.data.csv')
  parser.add_argument(
      '--eval-files',
      help='GCS file or local paths to evaluation data',
      nargs='+',
      default='gs://cloud-samples-data/ml-engine/census/data/adult.test.csv')
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      default='/tmp/census-estimator')
  parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      type=int)
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=40)
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=40)
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
      '--train-steps',
      help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.""",
      default=100,
      type=int)
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int)
  parser.add_argument(
      '--export-format',
      help='The input format of the exported SavedModel binary',
      choices=['JSON', 'CSV', 'EXAMPLE'],
      default='JSON')
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO')

  args, _ = parser.parse_known_args()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  train_and_evaluate(args)
