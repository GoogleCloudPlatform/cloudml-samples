# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Implements the vanilla tensorflow model on single node."""

# See https://goo.gl/JZ6hlH to contrast this with DNN combined
# which the high level estimator based sample implements.

import tensorflow as tf

def random_normal():
  """Random normal initializer."""
  return tf.random_normal_initializer(mean=0.0, stddev=0.1)

def create_layer(layer_name, weight_name, bias_name):
  with tf.variable_scope(layer_name):
    weight = tf.get_variable(weight_name,
                             shape,
                             initializer=random_normal())

    bias = tf.get_variable(bias_name,
                           shape[1],
                           initializer=tf.zeros_initializer(tf.float32))

def inference(input_x, hidden_units=[100,70,50,25], num_classes=2):
  """Create a Feed forward network running on single node

  Args:
    input_x (tf.placeholder): Feature placeholder input
    hidden_units (list): Hidden units
    num_classes (int): Number of classes
  """
  input_x = tf.to_float(input_x)
  previous_units = hidden_units[0]

  layers_size = [input_x.get_shape()[1]] + hidden_units
  layers_shape = zip(layers_size[0:],layers_size[1:])

  curr_layer = input_x

  # Creates the relu hidden layers
  for num, shape in enumerate(layers_shape):
    with tf.variable_scope("hidden_layer_{}".format(num)):
      weight = tf.get_variable("weight_{}".format(num),
                               shape,
                               initializer=random_normal())

      bias = tf.get_variable("bias_{}".format(num),
                             shape[1],
                             initializer=tf.zeros_initializer(tf.float32))

      curr_layer = tf.nn.relu(tf.matmul(curr_layer, weight) + bias)

  # Creates the softmax output layer
  # Shape of the weight matrix
  shape = [hidden_units[-1], num_classes]

  with tf.variable_scope("output_layer"):
    weight = tf.get_variable("weight_output",
                             shape,
                             initializer=random_normal())

    bias = tf.get_variable("bias_output",
                           shape[1],
                           initializer=tf.zeros_initializer(tf.float32))

    curr_layer = tf.nn.softmax(tf.matmul(curr_layer, weight) + bias)

  return curr_layer
