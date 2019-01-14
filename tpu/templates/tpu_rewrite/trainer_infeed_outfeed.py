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

# Reference: https://colab.research.google.com/gist/rjpower/169b2843a506d090f47d25122f82a28f


import argparse
from functools import partial
import numpy as np
import os
import threading

import tensorflow as tf
from tensorflow.contrib.cluster_resolver import TPUClusterResolver


def build_model(features):
    hidden = tf.layers.dense(features, 10, activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden, 1)

    return outputs


def fit_batch(features, labels):
    # inner function that specifies one step of calculation to be done on TPU.

    outputs = build_model(features)
    loss = tf.nn.l2_loss(outputs - labels)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)

    # Wrap the optimizer
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step) 

    return global_step, loss, train_op


def tpu_computation_with_infeed(batch_size, num_shards):
    # This function wrap around `fit_batch` and handles infeed/outfeed queues from the perspective of a TPU device.

    # The infeed queue is implicit and the tensors in it are not passed in as function arguments like in model_fn.
    features, labels = tf.contrib.tpu.infeed_dequeue_tuple(
        # the dtypes and shapes need to be consistent with what is fed into the infeed queue.
        dtypes=[tf.float32, tf.float32],
        shapes=[(batch_size // num_shards, 5), (batch_size // num_shards)]
    )

    global_step, loss, train_op = fit_batch(features, labels)

    # TPU functions must return zero-or more Tensor values followed by zero or more Operations.
    # The outfeed queue is also implicit.
    return tf.contrib.tpu.outfeed_enqueue_tuple((global_step, loss)), train_op


def setup_feed(features, labels, num_shards):
    # This function handles infeed/outfeed queues from the perspective of the CPU.
    infeed_ops = []
    outfeed_ops = []

    infeed_batches = zip(tf.split(features, num_shards), tf.split(labels, num_shards))

    for i, batch in enumerate(infeed_batches):
        infeed_op = tf.contrib.tpu.infeed_enqueue_tuple(
            batch,
            [b.shape for b in batch],
            device_ordinal=i
        )
        infeed_ops.append(infeed_op)

        outfeed_op = tf.contrib.tpu.outfeed_dequeue_tuple(
                dtypes=[tf.int64, tf.float32],
                shapes=[(), ()],
                device_ordinal=i
            )
        outfeed_ops.append(outfeed_op)

    return infeed_ops, outfeed_ops


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
    # Unpack the tensor batch to be used to set up the infeed/outfeed queues.
    dataset = train_input_fn()
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    infeed_ops, outfeed_ops = setup_feed(features, labels, num_shards=8)

    # Wrap the tpu computation function to be run in a loop.
    def computation_loop():
        return tf.contrib.tpu.repeat(args.max_steps, partial(tpu_computation_with_infeed, batch_size=16, num_shards=8))

    # Since we are using infeed/outfeed queues, tensors are not explicitly passed in or returned.
    tpu_computation_loop = tf.contrib.tpu.batch_parallel(computation_loop, num_shards=8)

    # utility ops
    tpu_init = tf.contrib.tpu.initialize_system()
    tpu_shutdown = tf.contrib.tpu.shutdown_system()
    variables_init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # get the TPU resource's grpc url
    # Note: when running on CMLE, args.tpu should be left as None
    tpu_grpc_url = TPUClusterResolver(tpu=args.tpu).get_master()
    sess = tf.Session(tpu_grpc_url)

    # Use separate threads to run infeed and outfeed.
    def _run_infeed():
        for i in range(args.max_steps):
            sess.run(infeed_ops)

            if i % args.save_checkpoints_steps == 0:
                print('infeed {}'.format(i))


    def _run_outfeed():
        for i in range(args.max_steps):
            outfeed_data = sess.run(outfeed_ops)

            if i % args.save_checkpoints_steps == 0:
                print('outfeed {}'.format(i))
                print('data returned from outfeed: {}'.format(outfeed_data))

                saver.save(sess, os.path.join(args.model_dir, 'model.ckpt'), global_step=i)


    infeed_thread = threading.Thread(target=_run_infeed)
    outfeed_thread = threading.Thread(target=_run_outfeed)

    sess.run(tpu_init)
    sess.run(variables_init)

    infeed_thread.start()
    outfeed_thread.start()

    sess.run(tpu_computation_loop)

    infeed_thread.join()
    outfeed_thread.join()

    sess.run(tpu_shutdown)

    saver.save(sess, os.path.join(args.model_dir, 'model.ckpt'), global_step=args.max_steps)


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
        help='The name or GRPC URL of the TPU node.  Leave it as `None` when training on CMLE.')

    args, _ = parser.parse_known_args()

    main(args)
