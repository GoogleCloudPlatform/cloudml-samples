#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
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

import argparse
from datetime import datetime
import logging
import tensorflow as tf
import os
import sys
import model
import experiment


# ******************************************************************************
# YOU MAY MODIFY THIS FUNCTION TO ADD/REMOVE PARAMS OR CHANGE THE DEFAULT VALUES
# ******************************************************************************


def get_args():
  """Define the task arguments with the default values.

  Returns:
      experiment parameters
  """

  args_parser = argparse.ArgumentParser()

  # Data files arguments
  args_parser.add_argument(
    '--train-files',
    help='GCS or local paths to training data.',
    nargs='+',
    required=True
  )
  args_parser.add_argument(
    '--eval-files',
    help='GCS or local paths to evaluation data.',
    nargs='+',
    required=True
  )

  ###########################################

  # Experiment arguments
  args_parser.add_argument(
    '--train-steps',
    help="""
      Steps to run the training job for. 
      If --num-epochs and --train-size are not specified, this must be. 
      Otherwise the training job will run indefinitely.
      if --num-epochs and --train-size are specified, 
      then --train-steps will be: (train-size/train-batch-size) * num-epochs
      """,
    default=1000,
    type=int
  )
  args_parser.add_argument(
    '--eval-steps',
    help="""
      Number of steps to run evaluation for at each checkpoint.',
      Set to None to evaluate on the whole evaluation data.
      """,
    default=None,
    type=int
  )
  args_parser.add_argument(
    '--batch-size',
    help='Batch size for each training and evaluation step.',
    type=int,
    default=128
  )
  args_parser.add_argument(
    '--train-size',
    help='Size of training set (instance count).',
    type=int,
    default=None
  )
  args_parser.add_argument(
    '--num-epochs',
    help="""
      Maximum number of training data epochs on which to train.
      If both --train-size and --num-epochs are specified,
      --train-steps will be: (train-size/train-batch-size) * num-epochs.
      """,
    default=None,
    type=int,
  )
  args_parser.add_argument(
    '--eval-frequency-secs',
    help='How many seconds to wait before running the next evaluation.',
    default=15,
    type=int
  )

  ###########################################

  # Feature columns arguments
  args_parser.add_argument(
    '--embed-categorical-columns',
    help="""
      If set to True, the categorical columns will be embedded 
      and used in the deep part og the model.
      The embedding size = sqrt(vocab_size).
      """,
    action='store_true',
    default=True,
  )
  args_parser.add_argument(
    '--use-indicator-columns',
    help="""
      If set to True, the categorical columns will be encoded 
      as One-Hot indicators in the deep part of the model.
      """,
    action='store_true',
    default=False,
  )
  args_parser.add_argument(
    '--use-wide-columns',
    help="""
      If set to True, the categorical columns will be used in the 
      wide part of the model.
      """,
    action='store_true',
    default=False,
  )
  ###########################################

  # Estimator arguments
  args_parser.add_argument(
    '--learning-rate',
    help="Learning rate value for the optimizers.",
    default=0.1,
    type=float
  )
  args_parser.add_argument(
    '--learning-rate-decay-factor',
    help="""
      The factor by which the learning rate should decay by the end of the training.
      decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps).
      If set to 1.0 (default), then no decay will occur.
      If set to 0.5, then the learning rate should reach 0.5 of its original value at the end of the training. 
      Note that decay_steps is set to train_steps.
      """,
    default=1.0,
    type=float
  )
  args_parser.add_argument(
    '--hidden-units',
    help="""
      Hidden layer sizes to use for DNN feature columns, provided in comma-separated layers. 
      If --scale-factor > 0, then only the size of the first layer will be used to compute 
      the sizes of subsequent layers.
      """,
    default='30,30,30'
  )
  args_parser.add_argument(
    '--layer-sizes-scale-factor',
    help="""
      Determine how the size of the layers in the DNN decays. 
      If value = 0 then the provided --hidden-units will be taken as is
      """,
    default=0.7,
    type=float
  )
  args_parser.add_argument(
    '--num-layers',
    help='Number of layers in the DNN. If --scale-factor > 0, then this parameter is ignored',
    default=4,
    type=int
  )
  args_parser.add_argument(
    '--dropout-prob',
    help="The probability we will drop out a given coordinate.",
    default=None
  )
  ###########################################

  # Saved model arguments
  args_parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True
  )
  args_parser.add_argument(
    '--reuse-job-dir',
    action='store_true',
    default=False,
    help="""
      Flag to decide if the model checkpoint should be re-used from the job-dir. 
      If set to False then the job-dir will be deleted.
      """
  )
  args_parser.add_argument(
    '--serving-export-format',
    help='The input format of the exported serving SavedModel.',
    choices=['JSON', 'CSV', 'EXAMPLE'],
    default='JSON'
  )
  args_parser.add_argument(
    '--eval-export-format',
    help='The input format of the exported evluating SavedModel.',
    choices=['CSV', 'EXAMPLE'],
    default='CSV'
  )
  ###########################################

  return args_parser.parse_args()

# ******************************************************************************
# THIS IS ENTRY POINT FOR THE TRAINER TASK
# ******************************************************************************


def _setup_logging():
  """Sets up logging."""
  root_logger = logging.getLogger()
  root_logger_previous_handlers = list(root_logger.handlers)
  for h in root_logger_previous_handlers:
    root_logger.removeHandler(h)
  root_logger.setLevel(logging.INFO)
  root_logger.propagate = False

  # Set tf logging to avoid duplicate logging. If the handlers are not removed,
  # then we will have duplicate logging
  tf_logger = logging.getLogger('TensorFlow')
  while tf_logger.handlers:
    tf_logger.removeHandler(tf_logger.handlers[0])

  # Redirect INFO logs to stdout
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.INFO)
  root_logger.addHandler(stdout_handler)

  # Suppress C++ level warnings.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
  args = get_args()
  _setup_logging()

  # If job_dir_reuse is False then remove the job_dir if it exists
  logging.info("Resume training:", args.reuse_job_dir)
  if not args.reuse_job_dir:
    if tf.gfile.Exists(args.job_dir):
      tf.gfile.DeleteRecursively(args.job_dir)
      logging.info("Deleted job_dir {} to avoid re-use".format(args.job_dir))
  else:
    logging.info("Reusing job_dir {} if it exists".format(args.job_dir))

  run_config = experiment.create_run_config(args)
  logging.info("Job directory:", run_config.model_dir)

  # Compute the number of training steps
  if args.train_size is not None and args.num_epochs is not None:
    args.train_steps = int(
      (args.train_size / args.batch_size) * args.num_epochs)
  else:
    args.train_steps = args.train_steps

  logging.info("Train size: {}.".format(args.train_size))
  logging.info("Epoch count: {}.".format(args.num_epochs))
  logging.info("Batch size: {}.".format(args.batch_size))
  logging.info("Training steps: {} ({}).".format(
    args.train_steps, "supplied" if args.train_size is None else "computed"))
  logging.info("Evaluate every {} steps.".format(args.eval_frequency_secs))

  # Create the estimator
  estimator = model.create(args, run_config)
  logging.info("creating an estimator: {}".format(type(estimator)))

  # Run the train and evaluate experiment
  time_start = datetime.utcnow()
  logging.info("Experiment started...")
  logging.info(".......................................")

  # Run experiment
  experiment.run(estimator, args)

  time_end = datetime.utcnow()
  logging.info(".......................................")
  logging.info("Experiment finished.")
  time_elapsed = time_end - time_start
  logging.info(
    "Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))


if __name__ == '__main__':
  main()
