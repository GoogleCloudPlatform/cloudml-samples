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


import os
import logging
import math
import tensorflow as tf
import tensorflow_model_analysis as tfma
import inputs


# ******************************************************************************
# YOU NEED NOT TO CHANGE THE FUNCTION TO RUN THE EXPERIMENT
# ******************************************************************************

def create_run_config(args):
  """Create a tf.estimator.RunConfig object.

  Args:
    args: experiment parameters.
  """

  # Configure the distribution strategy if GPUs available
  distribution_strategy = None
  # Get the available GPU devices
  num_gpus = len([device_name
                  for device_name in tf.contrib.eager.list_devices()
                  if '/device:GPU' in device_name])
  logging.info('%s GPUs are available.', str(num_gpus))
  if num_gpus > 1:
    distribution_strategy = tf.distribute.MirroredStrategy()
    logging.info('MirroredStrategy will be used for training.')
    # Update the batch size
    args.batch_size = int(math.ceil(args.batch_size / num_gpus))

  # Create RunConfig
  return tf.estimator.RunConfig(
    tf_random_seed=19831006,
    log_step_count_steps=100,
    model_dir=args.job_dir,
    save_checkpoints_secs=args.eval_frequency_secs,
    keep_checkpoint_max=3,
    train_distribute=distribution_strategy,
    eval_distribute=distribution_strategy
  )


def run(estimator, args):
  """Train, evaluate, and export the model for serving and evaluating.

  Args:
    estimator: TensorFlow Estimator.
    args: experiment parameters.
  """

  # Create TrainSpec
  train_spec = tf.estimator.TrainSpec(
    input_fn=inputs.make_input_fn(
      file_pattern=args.train_files,
      mode=tf.estimator.ModeKeys.TRAIN,
      batch_size=args.batch_size
    ),
    max_steps=int(args.train_steps)
  )

  # Create exporter for a serving model
  exporter = tf.estimator.FinalExporter(
    'estimate',
    inputs.SERVING_INPUT_RECEIVER_FUNCTIONS[args.serving_export_format]
  )

  # Create EvalSpec
  eval_spec = tf.estimator.EvalSpec(
    input_fn=inputs.make_input_fn(
      file_pattern=args.eval_files,
      mode=tf.estimator.ModeKeys.EVAL,
      batch_size=args.batch_size
    ),
    steps=args.eval_steps,
    exporters=[exporter],
    start_delay_secs=0,
    throttle_secs=0
  )

  # Train and evaluate
  logging.info("Training and evaluating...")
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  # Exporting the model for the TensorFlow Model Analysis
  logging.info("Exporting the model for evaluation...")
  tfma.export.export_eval_savedmodel(
    estimator=estimator,
    export_dir_base=os.path.join(estimator.model_dir, "export/evaluate"),
    eval_input_receiver_fn=inputs.EVALUATING_INPUT_RECEIVER_FUNCTIONS[
      args.eval_export_format]
  )
