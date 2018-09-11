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
import tensorflow as tf


def tpu_computation(features, labels):
    # build model
    hidden = tf.layers.dense(features, 10, activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden, 1)
    loss = tf.nn.l2_loss(outputs - labels)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)

    # Wrap the optimizer
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step) 

    # return the ops that the TPU needs to carry out
    return loss, train_op


def train_input_fn():
    # make some fake regression data
    x = np.random.rand(100, 5)
    w = np.random.rand(5)
    y = np.sum(x * w, axis=1)

    # TPUs currently do not support float64
    x_tensor = tf.constant(x, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.float32)

    # create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

    # COMMENT HERE!
    batch_size = 16

    dataset = dataset.repeat().shuffle(32).batch(batch_size, drop_remainder=True)

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
    features, labels = train_input_fn()

    train_on_tpu = tf.contrib.tpu.rewrite(tpu_computation, [features, labels])

    # FIXME: 'grpc://worker-address'
    sess = tf.Session('grpc://10.0.5.2')

    # FIXME: is this needed if running on CMLE?
    sess.run(tf.contrib.tpu.initialize_system())

    sess.run(tf.global_variables_initializer())

    for i in range(args.max_steps):
        loss, _ = sess.run(train_on_tpu)

        if i % 10 == 0:
            # TODO: add export model
            print(loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tpu-template'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--save-checkpoints-steps',
        type=int,
        default=100
    )
    parser.add_argument(
        '--use-tpu',
        action='store_true'
    )
    parser.add_argument(
        '--tpu',
        default=None
    )

    args, _ = parser.parse_known_args()

    main(args)
