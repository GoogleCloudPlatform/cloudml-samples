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

# Based on: https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/tpu/templates/tpu_rewrite/trainer_infeed_outfeed.py


import argparse
from collections import deque
from functools import partial
import numpy as np
import os
import random
import threading

import tensorflow as tf
from tensorflow.contrib.cluster_resolver import TPUClusterResolver


# Environment specific parameters.  Here we are using a fake environment.
FEATURE_SIZE = 128
ACTION_SIZE = 3
N_PARALLEL_GAMES = 16

# size of the experience
ROLLOUT_LENGTH = 1024
EXPERIENCE_LENGTH = ROLLOUT_LENGTH * 8

def policy(features):
    with tf.variable_scope('agent', reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(features, 200, activation=tf.nn.relu)
        logits = tf.layers.dense(hidden, ACTION_SIZE)

    return logits


def fit_batch(features, actions, rewards):
    # features are observations

    logits = policy(features)
    loss = rewards * tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions, logits=logits)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    return global_step, loss, train_op


def tpu_computation_with_infeed(batch_size, num_shards):
    # TPU device perspective.

    features, actions, rewards = tf.contrib.tpu.infeed_dequeue_tuple(
        # the dtypes and shapes need to be consistent with what is fed into the infeed queue.
        dtypes=[tf.float32, tf.int32, tf.float32],
        shapes=[
            (batch_size // num_shards, FEATURE_SIZE),
            (batch_size // num_shards, ACTION_SIZE),
            (batch_size // num_shards)
        ]
    )

    global_step, loss, train_op = fit_batch(features, actions, rewards)

    return tf.contrib.tpu.outfeed_enqueue_tuple((global_step, loss)), train_op


def cpu_setup_feed(features, actions, rewards, num_shards):
    # CPU perspective.

    infeed_ops = []
    outfeed_ops = []

    infeed_batches = zip(
        tf.split(features, num_shards),
        tf.split(actions, num_shards),
        tf.split(rewards, num_shards)
    )

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


def make_ds(v, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(v)
    dataset = dataset.repeat().shuffle(32).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    n_dim = len(next_batch.shape)
    merge_shape = [batch_size] + [None] * (n_dim - 1)
    shape = next_batch.shape.merge_with(merge_shape) 
    next_batch.set_shape(shape)

    return next_batch, iterator.initializer


def tf_deque(name, dtype, shape, update_size):
    variable = tf.get_variable(name, dtype=dtype, shape=shape, trainable=False)

    update_shape = [update_size] + shape[1:]
    update_ph = tf.placeholder(dtype=dtype, shape=update_shape)

    updated_value = tf.concat([v[update_size:], update_ph], axis=0)

    update_op = variable.assign(updated_value)

    return variable, update_ph, update_op


def main(args):
    # use variables to store experience
    features, update_features_ph, update_features_op = tf_deque('features', tf.float32, (EXPERIENCE_LENGTH, FEATURE_SIZE), ROLLOUT_LENGTH)
    actions, update_actions_ph, update_actions_op = tf_deque('actions', tf.int32, (EXPERIENCE_LENGTH, ACTION_SIZE), ROLLOUT_LENGTH)
    rewards, update_rewards_ph, update_rewards_op = tf_deque('rewards', tf.float32, (EXPERIENCE_LENGTH,), ROLLOUT_LENGTH)

    rollout_update_ops = [update_features_op, update_actions_op, update_rewards_op]

    features, features_init = make_ds(features, args.train_batch_size)
    actions, actions_init = make_ds(actions, args.train_batch_size)
    rewards, rewards_init = make_ds(rewards, args.train_batch_size)

    ds_inits = [features_init, actions_init, rewards_init]

    infeed_ops, outfeed_ops = cpu_setup_feed(features, actions, rewards, num_shards=8)

    # Wrap the tpu computation function to be run in a loop.
    def computation_loop():
        return tf.contrib.tpu.repeat(args.iterations_per_loop, partial(tpu_computation_with_infeed, batch_size=16, num_shards=8))

    tpu_computation_loop = tf.contrib.tpu.batch_parallel(computation_loop, num_shards=8)

    # CPU policy used for interacting with the environment
    features_ph = tf.placeholder(dtype=tf.float32, shape=(N_PARALLEL_GAMES, FEATURE_SIZE))
    rollout_logits = policy(features_ph)
    rollout_actions = tf.one_hot(tf.argmax(rollout_logits, axis=-1), depth=ACTION_SIZE)

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


    # In the main thread, interact with th environment and collect data into the experience variables.
    def run_rollout():
        episode_length = ROLLOUT_LENGTH // N_PARALLEL_GAMES
        
        batch_features = []
        batch_actions = []
        batch_rewards = []

        while len(batch_features) < MAXLEN:
            # Randomly generate features, not feeding the actions to the environment.
            step_features = np.random.random((N_PARALLEL_GAMES, FEATURE_SIZE))

            # Since the CPU and the TPU share the model variables, this is using the updated policy.
            step_actions = sess.run(rollout_actions, {features_ph: step_features})

            step_rewards = np.random.random((N_PARALLEL_GAMES,))

            batch_features.extend(step_features.tolist())
            batch_actions.extend(step_actions)
            batch_rewards.extend(step_rewards.tolist())

        rollout_feed_dict = {
            update_features_ph: np.array(batch_features),
            update_actions_ph: np.array(batch_actions),
            update_rewards_ph: np.array(batch_rewards)
        }
        sess.run(rollout_update_ops, rollout_feed_dict)

    infeed_thread = threading.Thread(target=_run_infeed)
    outfeed_thread = threading.Thread(target=_run_outfeed)

    sess.run(tpu_init)
    sess.run(variables_init)
    sess.run(ds_inits)

    run_rollout()

    infeed_thread.start()
    outfeed_thread.start()

    for i in range(args.num_loops):
        print('Iteration: {}'.format(i))

        run_rollout()
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
        '--iterations-per-loop',
        type=int,
        default=100,
        help='The number of iterations on TPU before switching to CPU.')
    parser.add_argument(
        '--num-loops',
        type=int,
        default=10,
        help='The number of times switching to CPU.')
    parser.add_argument(
        '--save-checkpoints-steps',
        type=int,
        default=100,
        help='The number of training steps before saving each checkpoint.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=16,
        help='The training batch size.  The training batch is divided evenly across the TPU cores.')
    parser.add_argument(
        '--tpu',
        default=None,
        help='The name or GRPC URL of the TPU node.  Leave it as `None` when training on CMLE.')

    args, _ = parser.parse_known_args()

    args.max_steps = args.iterations_per_loop * args.num_loops

    main(args)
