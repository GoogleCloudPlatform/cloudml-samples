#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

LEARNING_RATE = 1e-4


def create_model(data_format):
	"""Model to recognize digits in the MNIST dataset.

	Network structure is equivalent to:
	https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples
	/tutorials/mnist/mnist_deep.py
	and
	https://github.com/tensorflow/models/blob/master/tutorials/image/mnist
	/convolutional.py

	But uses the tf.keras API.

	Args:
		data_format: Either 'channels_first' or 'channels_last'.
		'channels_first' is
			typically faster on GPUs while 'channels_last' is typically faster on
			CPUs. See
			https://www.tensorflow.org/performance/performance_guide#data_formats

	Returns:
		A tf.keras.Model.
	"""
	if data_format == 'channels_first':
		input_shape = [1, 28, 28]
	else:
		assert data_format == 'channels_last'
		input_shape = [28, 28, 1]

	l = tf.keras.layers
	max_pool = l.MaxPooling2D(
		(2, 2), (2, 2), padding='same', data_format=data_format)
	# The model consists of a sequential chain of layers, so tf.keras.Sequential
	# (a subclass of tf.keras.Model) makes for a compact description.
	return tf.keras.Sequential(
		[
			l.Reshape(
				target_shape=input_shape,
				input_shape=(28 * 28,)),
			l.Conv2D(
				32,
				5,
				padding='same',
				data_format=data_format,
				activation=tf.nn.relu),
			max_pool,
			l.Conv2D(
				64,
				5,
				padding='same',
				data_format=data_format,
				activation=tf.nn.relu),
			max_pool,
			l.Flatten(),
			l.Dense(1024, activation=tf.nn.relu),
			l.Dropout(0.4),
			l.Dense(10)
		])


def model_fn(features, labels, mode, params):
	"""The model_fn argument for creating an Estimator."""
	model = create_model(params['data_format'])
	image = features
	if isinstance(image, dict):
		image = features['image']

	# Training
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

		logits = model(image, training=True)
		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		accuracy = tf.metrics.accuracy(
			labels=labels, predictions=tf.argmax(logits, axis=1))

		# Name tensors to be logged with LoggingTensorHook.
		tf.identity(LEARNING_RATE, 'learning_rate')
		tf.identity(loss, 'cross_entropy')
		tf.identity(accuracy[1], name='train_accuracy')

		# Save accuracy scalar to Tensorboard output.
		tf.summary.scalar('train_accuracy', accuracy[1])

		return tf.estimator.EstimatorSpec(
			mode=tf.estimator.ModeKeys.TRAIN,
			loss=loss,
			train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

	# Evaluate
	if mode == tf.estimator.ModeKeys.EVAL:
		logits = model(image, training=False)
		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		return tf.estimator.EstimatorSpec(
			mode=tf.estimator.ModeKeys.EVAL,
			loss=loss,
			eval_metric_ops={
				'accuracy':
					tf.metrics.accuracy(
						labels=labels, predictions=tf.argmax(logits, axis=1)),
			})

	# Predict
	if mode == tf.estimator.ModeKeys.PREDICT:
		logits = model(image, training=False)
		predictions = {
			'classes': tf.argmax(logits, axis=1),
			'probabilities': tf.nn.softmax(logits),
		}
		return tf.estimator.EstimatorSpec(
			mode=tf.estimator.ModeKeys.PREDICT,
			predictions=predictions,
			export_outputs={
				'classify': tf.estimator.export.PredictOutput(predictions)
			})