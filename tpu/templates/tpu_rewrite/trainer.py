# Copyright 2018 Google LLC
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


import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

def tpu_computation(features, labels):
    # Similar to the role of model_fn, the TPU function builds the part of the graph to be run on TPUs

    # build model
    hidden = tf.layers.dense(features, 10, activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden, 1)
    loss = tf.nn.l2_loss(outputs - labels)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)

    # Wrap the optimizer
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step) 

    # TPU functions must return zero-or more Tensor values followed by zero or more Operations.
    return global_step, loss, train_op


def train_input_fn():
    # data input function runs on the CPU, not TPU

    # make some fake regression data
    x = np.random.rand(100, 5)
    w = np.random.rand(5)
    y = np.sum(x * w, axis=1)

    # TPUs currently do not support float64
    x_tensor = tf.constant(x, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.float32)

    # create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

    # TPUs need to know all dimensions including batch size
    batch_size = 16

    dataset = dataset.repeat().shuffle(32).batch(batch_size)#, drop_remainder=True)

    # TPUs need to know all dimensions when the graph is built
    # Datasets know the batch size only when the graph is run
    def set_shapes(features, labels):
        features_shape = features.get_shape().merge_with([batch_size, None])
        labels_shape = labels.get_shape().merge_with([batch_size])

        features.set_shape(features_shape)
        labels.set_shape(labels_shape)

        return features, labels

    dataset = dataset.map(set_shapes)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def main(args):
    # unpack the tensor batch to be used as the list of inputs of the TPU function
    dataset = train_input_fn()
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    # mark part of the graph to be run on the TPUs
    global_step_tensor, loss_tensor = tf.contrib.tpu.rewrite(tpu_computation, [features, labels])

    # utility ops
    tpu_init = tf.contrib.tpu.initialize_system()
    tpu_shutdown = tf.contrib.tpu.shutdown_system()
    variables_init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # get the TPU resource's grpc url
    # Note: when running on AI Platform, args.tpu should be left as None
    tpu_grpc_url = TPUClusterResolver(tpu=args.tpu).get_master()
    sess = tf.Session(tpu_grpc_url)

    sess.run(tpu_init)
    sess.run(variables_init)

    for i in range(args.max_steps):
        # the tensor values in the TPU function are returned in a list, and the operations in the TPU function are called with no return value
        global_step, loss = sess.run([global_step_tensor, loss_tensor])

        if i % args.save_checkpoints_steps == 0:
            saver.save(sess, os.path.join(args.model_dir, 'model.ckpt'), global_step=global_step)

            tf.logging.info('global_step: {}, loss: {}'.format(global_step, loss))

    sess.run(tpu_shutdown)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tpu-template',
        help='Location to write checkpoints and summaries to.  Must be a GCS URI when using Cloud TPU.')
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000,
        help='The total number of steps to train the model.')
    parser.add_argument(
        '--save-checkpoints-steps',
        type=int,
        default=100,
        help='The number of training steps before saving each checkpoint.')
    parser.add_argument(
        '--tpu',
        default=None,
        help='The name or GRPC URL of the TPU node.  Leave it as `None` when training on AI Platform.')

    args, _ = parser.parse_known_args()

    main(args)
