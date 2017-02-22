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

def inference(x, hidden_units=[100,70,50,25], y_units=2):
  """Create a Feed forward network running on single node

  Args:
    x: Feature placeholder input
    hidden_units: Hidden units
    y_units: Number of classes
  """
  x = tf.to_float(x)
  previous_units = hidden_units[0]

  layers_size = [x.get_shape()[1]] + hidden_units + [y_units]
  layers_shape = zip(layers_size[0:],layers_size[1:])

  curr_layer = x
  for num, shape in enumerate(layers_shape):
    with tf.variable_scope("layer_{}".format(num)):
      weight = tf.get_variable("weight_{}".format(num),
                               shape)
      bias = tf.get_variable("bias_{}".format(num),
                             shape[1],
                             initializer=tf.zeros_initializer(tf.float32))

      if num < len(layers_shape) - 1:
        curr_layer = tf.nn.relu(tf.add(tf.matmul(curr_layer, weight), bias))
      else:
        curr_layer = tf.nn.softmax(tf.add(tf.matmul(curr_layer, weight), bias))

  return curr_layer
