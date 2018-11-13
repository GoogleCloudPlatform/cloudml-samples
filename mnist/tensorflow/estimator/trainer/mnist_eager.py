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
"""MNIST model training with TensorFlow eager execution.

See:
https://research.googleblog.com/2017/10/eager-execution-imperative-define-by
.html

This program demonstrates training of the convolutional neural network model
defined in mnist.py with eager execution enabled.

If you are not interested in eager execution, you should ignore this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing
import time
import os

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
# pylint: enable=g-bad-import-order


from . import dataset as mnist_dataset
from . import model as mnist_model
from . import model_helpers

tfe = tf.contrib.eager


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
		default='/tmp/tensorflow/mnist/input_data',
		help='The location of the input data directory.')
	parser.add_argument(
		'--model_dir',
		type=str,
		required=True,
		default='/tmp/tensorflow/mnist/checkpoints/',
		help='The location of the model checkpoint files.')
	parser.add_argument(
		'--clean',
		type=lambda x: (str(x).lower() == 'true'),
		default=False,
		help='If set, model_dir will be removed if it exists.')
	parser.add_argument(
		'--train_epochs',
		type=int,
		default=10,
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
				 'trigger the end of training')
	parser.add_argument(
		'--batch_size',
		type=int,
		default=100,
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
		'--log_interval',
		type=int,
		default=10,
		help='Batches between logging training status.')
	parser.add_argument(
		'--output_dir',
		type=str,
		help='Directory to write TensorBoard summaries.')
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=0.01,
		help='Learning rate.')
	parser.add_argument(
		'--momentum',
		type=float,
		default=0.5,
		help='SGD momentum.')
	parser.add_argument(
		'--no_gpu',
		type=lambda x: (str(x).lower() == 'true'),
		default=False,
		help='If set, model_dir will be removed if it exists.')
	parser.add_argument(
		'--verbosity',
		choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
		default='INFO')

	return parser.parse_args()


def loss(logits, labels):
	return tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=logits, labels=labels))


def compute_accuracy(logits, labels):
	predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
	labels = tf.cast(labels, tf.int64)
	batch_size = int(logits.shape[0])
	return tf.reduce_sum(
		tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def train(model, optimizer, dataset, step_counter, log_interval=None):
	"""Trains model on `dataset` using `optimizer`."""

	start = time.time()
	for (batch, (images, labels)) in enumerate(dataset):
		with tf.contrib.summary.record_summaries_every_n_global_steps(
			10, global_step=step_counter):
			# Record the operations used to compute the loss given the input,
			# so that the gradient of the loss with respect to the variables
			# can be computed.
			with tf.GradientTape() as tape:
				logits = model(images, training=True)
				loss_value = loss(logits, labels)
				tf.contrib.summary.scalar('loss', loss_value)
				tf.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))
			grads = tape.gradient(loss_value, model.variables)
			optimizer.apply_gradients(
				zip(grads, model.variables), global_step=step_counter)
			if log_interval and batch % log_interval == 0:
				rate = log_interval / (time.time() - start)
				print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value,
																											 rate))
				start = time.time()


def test(model, dataset):
	"""Perform an evaluation of `model` on the examples from `dataset`."""
	avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
	accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

	for (images, labels) in dataset:
		logits = model(images, training=False)
		avg_loss(loss(logits, labels))
		accuracy(
			tf.argmax(logits, axis=1, output_type=tf.int64),
			tf.cast(labels, tf.int64))
	print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
				(avg_loss.result(), 100 * accuracy.result()))
	with tf.contrib.summary.always_record_summaries():
		tf.contrib.summary.scalar('loss', avg_loss.result())
		tf.contrib.summary.scalar('accuracy', accuracy.result())


def train_and_evaluate(hparams):
	"""Run MNIST training and eval loop in eager mode.

	Args:
		hparams: An object containing parsed flag values.
	"""
	tf.enable_eager_execution()
	model_helpers.apply_clean(hparams)

	# Automatically determine device and data_format
	(device, data_format) = ('/gpu:0', 'channels_first')
	if hparams.no_gpu or not tf.test.is_gpu_available():
		(device, data_format) = ('/cpu:0', 'channels_last')
	# If data_format is defined in hparams, overwrite automatically set value.
	if hparams.data_format is not None:
		data_format = hparams.data_format
	print('Using device %s, and data format %s.' % (device, data_format))

	# Load the datasets.
	train_ds = mnist_dataset.train(hparams.data_dir).shuffle(60000).batch(
		hparams.batch_size)
	test_ds = mnist_dataset.test(hparams.data_dir).batch(
		hparams.batch_size)

	# Create the model and optimizer.
	_model = mnist_model.create_model(data_format)
	optimizer = tf.train.MomentumOptimizer(hparams.learning_rate,
																				 hparams.momentum)

	# Create file writers for writing TensorBoard summaries.
	if hparams.output_dir:
		# Create directories to which summaries will be written
		# tensorboard --logdir=<output_dir>
		# can then be used to see the recorded summaries.
		train_dir = os.path.join(hparams.output_dir, 'train')
		test_dir = os.path.join(hparams.output_dir, 'eval')
		tf.gfile.MakeDirs(hparams.output_dir)
	else:
		train_dir = None
		test_dir = None
	summary_writer = tf.contrib.summary.create_file_writer(
		train_dir, flush_millis=10000)
	test_summary_writer = tf.contrib.summary.create_file_writer(
		test_dir, flush_millis=10000, name='test')

	# Create and restore checkpoint (if one exists on the path)
	checkpoint_prefix = os.path.join(hparams.model_dir, 'ckpt')
	step_counter = tf.train.get_or_create_global_step()
	checkpoint = tf.train.Checkpoint(
		model=_model, optimizer=optimizer, step_counter=step_counter)
	# Restore variables on creation if a checkpoint exists.
	checkpoint.restore(tf.train.latest_checkpoint(hparams.model_dir))

	# Train and evaluate for a set number of epochs.
	with tf.device(device):
		for _ in range(hparams.train_epochs):
			start = time.time()
			with summary_writer.as_default():
				train(_model, optimizer, train_ds, step_counter,
							hparams.log_interval)
			end = time.time()
			print('\nTrain time for epoch #%d (%d total steps): %f' %
						(checkpoint.save_counter.numpy() + 1,
						 step_counter.numpy(),
						 end - start))
			with test_summary_writer.as_default():
				test(_model, test_ds)
			checkpoint.save(checkpoint_prefix)


if __name__ == '__main__':
	args = get_args()
	tf.logging.set_verbosity(args.verbosity)

	hparams = hparam.HParams(**args.__dict__)
	train_and_evaluate(hparams)
