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
import multiprocessing
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

from . import dataset
from . import distribution_utils
from . import model
from . import hooks_helper
from . import model_helpers


def get_num_gpus(hparams):
	"""Treat num_gpus=-1 as 'use all'."""
	if hparams.num_gpus != -1:
		return hparams.num_gpus

	from tensorflow.python.client import \
		device_lib  # pylint: disable=g-import-not-at-top
	local_device_protos = device_lib.list_local_devices()
	return sum([1 for d in local_device_protos if d.device_type == "GPU"])


def get_args():
	"""Argument parser.

	Returns:
		Dictionary of arguments.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data_dir',
		type=str,
		required=True,
		default='/tmp',
		help='The location of the input data directory.')
	parser.add_argument(
		'--model_dir',
		type=str,
		required=True,
		default='/tmp',
		help='The location of the model checkpoint files.')
	parser.add_argument(
		'--clean',
		type=lambda x: (str(x).lower() == 'true'),
		default=False,
		help='If set, model_dir will be removed if it exists.')
	parser.add_argument(
		'--train_epochs',
		type=int,
		default=1,
		help='The number of epochs used to train.')
	parser.add_argument(
		'--epochs_between_evals',
		type=int,
		default=1,
		help='The number of training epochs to run between evaluations.')
	parser.add_argument(
		'--stop_threshold',
		type=float,
		help='Specify a threshold accuracy or other eval metric which should '
				 'trigger the end of training', )
	parser.add_argument(
		'--batch_size',
		type=int,
		default=32,
		help='Batch size for training and evaluation. When using multiple gpus, '
				 'this is the global batch size for all devices. For example, if the '
				 'batch size is 32 and there are 4 GPUs, each GPU will get 8 examples '
				 'on each step.')
	parser.add_argument(
		'--num_gpus',
		type=int,
		default=1 if tf.test.is_gpu_available() else 0,
		help='How many GPUs to use with the DistributionStrategies API. The '
				 'default is 1 if TensorFlow can detect a GPU, and 0 otherwise.')
	parser.add_argument(
		'--hooks',
		type=str,
		nargs='+',
		default=['LoggingTensorHook'],
		help='A list of (case insensitive) strings to specify the names of '
				 'training hooks. Example: `--hooks ProfilerHook, '
				 'ExamplesPerSecondHook`. See official.utils.logs.hooks_helper for '
				 'details')
	parser.add_argument(
		'--export_dir',
		type=str,
		help='If set, a SavedModel serialization of the model will be exported to '
				 'this directory at the end of training. See the README for more '
				 'details and relevant links.')
	# Performance tuning arguments.
	parser.add_argument(
		'--num_parallel_calls',
		type=int,
		default=multiprocessing.cpu_count(),
		help='The number of records that are  processed in parallel during input '
				 'processing. This can be optimized per data set but for generally '
				 'homogeneous data sets, should be approximately the number of '
				 'available CPU cores. (default behavior)')
	parser.add_argument(
		'--inter_op_parallelism_threads',
		type=int,
		default=0,
		help='Number of inter_op_parallelism_threads to use for CPU. See '
				 'TensorFlow config.proto for details.')
	parser.add_argument(
		'--intra_op_parallelism_threads',
		type=int,
		default=0,
		help='Number of intra_op_parallelism_threads to use for CPU. See '
				 'TensorFlow config.proto for details.')
	parser.add_argument(
		'--use_synthetic_data',
		type=lambda x: (str(x).lower() == 'true'),
		default=False,
		help='If set, use fake data (zeroes) instead of a real dataset.')
	parser.add_argument(
		'--max_train_steps',
		type=int,
		help='The model will stop training if the global_step reaches this '
				 'value. If not set, training will run until the specified number of '
				 'epochs have run as usual. It is generally recommended to set '
				 '--train_epochs=1 when using this flag.')
	parser.add_argument(
		'--all_reduce_alg',
		type=str,
		help='Defines the algorithm to use for performing all-reduce '
				 'See tf.contrib.distribute.AllReduceCrossTowerOps for '
				 'more details and available options.')
	parser.add_argument(
		'--data_format',
		choices=['channels_first', 'channels_last'],
		help='A flag to override the data format used in the model. '
				 'channels_first '
				 'provides a performance boost on GPU but is not always compatible '
				 'with CPU. If left unspecified, the data format will be chosen '
				 'automatically based on whether TensorFlow was built for CPU or GPU.')
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
	model_function = model.model_fn
	session_config = tf.ConfigProto(
		inter_op_parallelism_threads=hparams.inter_op_parallelism_threads,
		intra_op_parallelism_threads=hparams.intra_op_parallelism_threads,
		allow_soft_placement=True)

	distribution_strategy = distribution_utils.get_distribution_strategy(
		get_num_gpus(hparams), hparams.all_reduce_alg)

	run_config = tf.estimator.RunConfig(
		train_distribute=distribution_strategy, session_config=session_config)

	data_format = hparams.data_format
	if data_format is None:
		data_format = ('channels_first'
									 if tf.test.is_built_with_cuda() else 'channels_last')
	mnist_classifier = tf.estimator.Estimator(
		model_fn=model_function,
		model_dir=hparams.model_dir,
		config=run_config,
		params={
			'data_format': data_format,
		})

	# Set up training and evaluation input functions.
	def train_input_fn():
		"""Prepare data for training."""

		# When choosing shuffle buffer sizes, larger sizes result in better
		# randomness, while smaller sizes use less memory. MNIST is a small
		# enough dataset that we can easily shuffle the full epoch.
		ds = dataset.train(hparams.data_dir)
		ds = ds.cache().shuffle(buffer_size=50000).batch(hparams.batch_size)

		# Iterate through the dataset a set number (`epochs_between_evals`) of
		# times during each training session.
		ds = ds.repeat(hparams.epochs_between_evals)
		return ds

	def eval_input_fn():
		return dataset.test(hparams.data_dir).batch(
			hparams.batch_size).make_one_shot_iterator().get_next()

	# Set up hook that outputs training logs every 100 steps.
	train_hooks = hooks_helper.get_train_hooks(
		hparams.hooks, model_dir=hparams.model_dir,
		batch_size=hparams.batch_size)

	# Train and evaluate model.
	for _ in range(hparams.train_epochs // hparams.epochs_between_evals):
		mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
		eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
		print('\nEvaluation results:\n\t%s\n' % eval_results)

		if model_helpers.past_stop_threshold(hparams.stop_threshold,
																				 eval_results['accuracy']):
			break

	# Export the model.
	if hparams.export_dir is not None:
		image = tf.placeholder(tf.float32, [None, 28, 28])
		input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
			'image': image,
		})
		mnist_classifier.export_savedmodel(hparams.export_dir, input_fn,
																			 strip_default_attrs=True)


if __name__ == '__main__':
	args = get_args()
	tf.logging.set_verbosity(args.verbosity)

	hparams = hparam.HParams(**args.__dict__)
	train_and_evaluate(hparams)
