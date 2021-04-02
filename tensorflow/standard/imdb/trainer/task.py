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
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

from . import model
from . import utils


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
		'--word-index-file',
		type=str,
		required=True,
		help='Word index json file local or GCS')
	parser.add_argument(
		'--num-epochs',
		type=float,
		default=40,
		help='number of times to go through the data, default=40')
	parser.add_argument(
		'--batch-size',
		default=512,
		type=int,
		help='number of records to read during each training step, default=512', )
	parser.add_argument(
		'--learning-rate',
		default=.001,
		type=float,
		help='Learning rate for gradient descent, default=.001')
	parser.add_argument(
		'--verbosity',
		choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
		default='INFO')
	
	return parser.parse_args()


def train_and_evaluate(hparams):
	"""Helper function: Trains and evaluate model.

	Args:
		hparams: (hparam.HParams) Command line parameters passed from task.py
	"""
	# Load data.
	(train_data, train_labels), (test_data, test_labels) = \
		utils.preprocess(train_data_file=hparams.train_file,
                         word_index_file=hparams.word_index_file,
                         num_words=model.TOP_K)

	# Training steps
	train_steps = hparams.num_epochs * len(train_data) / hparams.batch_size
	# Change numpy array shape.
	train_labels = np.asarray(train_labels).astype('int').reshape((-1, 1))
	# Create TrainSpec.
	train_spec = tf.estimator.TrainSpec(
		input_fn=lambda: model.input_fn(
			train_data,
			train_labels,
			hparams.batch_size,
			mode=tf.estimator.ModeKeys.TRAIN),
		max_steps=train_steps)
	# Create exporter information.
	exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
	# Change numpy array shape.
	test_labels = np.asarray(test_labels).astype('int').reshape((-1, 1))
	# Create EvalSpec.
	eval_spec = tf.estimator.EvalSpec(
		input_fn=lambda: model.input_fn(
			test_data,
			test_labels,
			hparams.batch_size,
			mode=tf.estimator.ModeKeys.EVAL),
		steps=None,
		exporters=exporter,
		start_delay_secs=10,
		throttle_secs=10)

	# Generate Configuration.
	run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
	# Create estimator
	estimator = model.keras_estimator(model_dir=hparams.job_dir,
                                      config=run_config,
                                      learning_rate=hparams.learning_rate,
                                      vocab_size=model.VOCAB_SIZE)
	# Start training
	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
	
	args = get_args()
	tf.logging.set_verbosity(args.verbosity)
	
	hparams = hparam.HParams(**args.__dict__)
	train_and_evaluate(hparams)
