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


def dnn(inputs, hidden_units=[100,70,50,25], num_classes=2):
  """Create a Feed forward network running on single node

  Args:
    input_x (tf.placeholder): Feature placeholder input
    hidden_units (list): Hidden units
    num_classes (int): Number of classes

  Returns:
    The last layer values
  """
  print(inputs.get_shape())
  layers_size = [inputs.get_shape()[1]] + hidden_units
  layers_shape = zip(layers_size[0:],layers_size[1:] + [num_classes])

  curr_layer = inputs
  with tf.variable_scope('dnn',
                         initializer=tf.truncated_normal_initializer()):
    # Creates the relu hidden layers
    for num, shape in enumerate(layers_shape):
      with tf.variable_scope('relu_{}'.format(num)):

        weights = tf.get_variable('weights', shape)

        biases = tf.get_variable(
            'biases', shape[1], initializer=tf.zeros_initializer(tf.float32))

      curr_layer = tf.nn.relu(tf.matmul(curr_layer, weights) + biases)

    return curr_layer



