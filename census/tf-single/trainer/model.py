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

def random_normal():
  """Return random normalizer intializer."""
  return tf.random_normal_initializer(mean=0.0, stddev=0.1)


def inference(x, distributed, hidden_units=[100,70,50,25], y_units=2):
  """Return a single or distributed graph."""
  if distributed:
    return inference_distributed(x, hidden_units, y_units)
  else:
    return inference_single(x, hidden_units, y_units)

def inference_single(x, hidden_units, y_units):
  """Create a Feed forward network running on single node

  Args:
    x: Feature placeholder input
    hidden_units: Hidden units
    y_units: Number of classes
  """
  x = tf.to_float(x)
  previous_units = hidden_units[0]

  with tf.variable_scope("layer_0") as scope:
    weight = tf.get_variable("weight_0",
                             [x.get_shape()[1], previous_units],
                             initializer=random_normal())

    bias = tf.get_variable("bias_0",
                           [previous_units],
                           initializer=random_normal())

    hidden_layer = tf.add(tf.matmul(x, weight), bias)
    hidden_layer = tf.nn.relu(hidden_layer)

  for layer_num, units in enumerate(hidden_units[1:]):
    layer_num+=1
    with tf.variable_scope("layer_{}".format(layer_num)) as scope:
      weight = tf.get_variable("weight_{}".format(layer_num),
                               [previous_units, units],
                               initializer=random_normal())

      bias = tf.get_variable("bias_{}".format(layer_num),
                             [units],
                             initializer=random_normal())

      layer = tf.add(tf.matmul(hidden_layer, weight), bias)
      layer = tf.nn.relu(layer)

      previous_units = units
      hidden_layer = layer

  layer_num+=1

  with tf.variable_scope("layer_{}".format(layer_num)) as scope:
    weight = tf.get_variable("weight_{}".format(layer_num),
                             [previous_units, y_units],
                             initializer=random_normal())

    bias = tf.get_variable("bias_{}".format(layer_num),
                           [y_units],
                           initializer=random_normal())

    output_layer = tf.add(tf.matmul(hidden_layer, weight), bias)
    output_layer = tf.nn.softmax(output_layer)

  return output_layer

def to_cluster_spec(tf_config):

def inference_distributed(x, hidden_units, y_units, cluster_spec):
  server = tf.train.Server(cluster_spec, job_name=name, task_index=index)
  x = tf.to_float(x)
  return None
