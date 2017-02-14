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

# See https://goo.gl/JZ6hlH for DNN combined

import tensorflow as tf

def init_variable(shape):
  """Initialize the weights."""
  weights = tf.random_normal(shape, stddev=0.1)
  return tf.Variable(weights)

def build_train_graph(input_tensors):
  return None


def build_eval_graph(input_tensors):
  return None


def inference(x, hidden_units=[100,70,50,25], y_units=2):
  """Create a Feed forward network with hidden units ReLU and SoftMax.

  Args:
    x: Feature placeholder input
    hidden_units: Hidden units
    y_units: Number of classes
  """
  x = tf.to_float(x)
  previous_units = hidden_units[0]
  weight = init_variable([x.get_shape()[1].value, previous_units])
  bias = init_variable([previous_units])
  hidden_layer = tf.add(tf.matmul(x, weight), bias)
  hidden_layer = tf.nn.relu(hidden_layer)

  for units in hidden_units[1:]:
    weight = init_variable([previous_units, units])
    bias = init_variable([units])

    layer = tf.add(tf.matmul(hidden_layer, weight), bias)
    layer = tf.nn.relu(layer)

    previous_units = units
    hidden_layer = layer

  weight = init_variable([previous_units, y_units])
  bias = init_variable([y_units])
  output_layer = tf.add(tf.matmul(hidden_layer, weight), bias)
  output_layer = tf.nn.softmax(output_layer)

  return output_layer
