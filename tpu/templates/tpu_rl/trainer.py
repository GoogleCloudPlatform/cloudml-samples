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

# Store (observation, onehot label of action, processed reward) tuples
MAXLEN = 1024

EXPERIENCE = deque([], maxlen=1000)

def experience_generator():
    while True:
        yield random.choice(list(EXPERIENCE))


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


def train_input_fn():
    # data input function runs on the CPU, not TPU

    dataset = tf.data.Dataset.from_generator(experience_generator, output_types=(tf.float32, tf.int32, tf.float32))

    batch_size = 16

    dataset = dataset.repeat().shuffle(32).batch(batch_size)

    # TPUs need to know all dimensions when the graph is built
    # Datasets know the batch size only when the graph is run
    def set_shapes(features, actions, rewards):
        features_shape = [batch_size, FEATURE_SIZE]
        actions_shape = [batch_size, ACTION_SIZE]
        rewards_shape = [batch_size]

        features.set_shape(features_shape)
        actions.set_shape(actions_shape)
        rewards.set_shape(rewards_shape)

        return features, actions, rewards

    dataset = dataset.map(set_shapes)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def make_ds(v, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(v)
    ds = ds.repeat().shuffle(32).batch(batch_size)
    it = ds.make_initializable_iterator()
    nb = it.get_next()

    return nb, it.initializer


def main(args):
    # Unpack the tensor batch to be used to set up the infeed/outfeed queues.
    # dataset = train_input_fn()
    # iterator = dataset.make_one_shot_iterator()
    # features, actions, rewards = iterator.get_next()

    # use variables to store experience
    features = tf.get_variable('features', dtype=tf.float32, shape=(MAXLEN, FEATURE_SIZE), trainable=False)
    actions = tf.get_variable('actions', dtype=tf.int32, shape=(MAXLEN, ACTION_SIZE), trainable=False)
    rewards = tf.get_variable('rewards', dtype=tf.float32, shape=(MAXLEN,), trainable=False)

    rollout_features_ph = tf.placeholder(dtype=tf.float32, shape=(MAXLEN, FEATURE_SIZE))
    rollout_actions_ph = tf.placeholder(dtype=tf.int32, shape=(MAXLEN, ACTION_SIZE))
    rollout_rewards_ph = tf.placeholder(dtype=tf.float32, shape=(MAXLEN,))

    rollout_update_ops = [
        features.assign(rollout_features_ph),
        actions.assign(rollout_actions_ph),
        rewards.assign(rollout_rewards_ph)
    ]

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


    # In the main thread, interact with th environment and collect data into EXPERIENCE.
    def run_rollout():
        episode_length = MAXLEN // N_PARALLEL_GAMES
        
        rollout_features = []
        rollout_actions = []
        rolloout_rewards = []

        while len(rollout_features) < MAXLEN:
            # Randomly generate features, not feeding the actions to the environment.
            step_features = np.random.random((N_PARALLEL_GAMES, FEATURE_SIZE))

            # Since the CPU and the TPU share the model variables, this is using the updated policy.
            step_actions = sess.run(rollout_actions, {features_ph: step_features})

            step_rewards = np.random.random((N_PARALLEL_GAMES, 1))

            rollout_features.extend(step_features.tolist())
            rollout_actions.extend(step_actions.tolist())
            rollout_rewards.extend(step_rewards.tolist())

        rollout_feed_dict = {
            rollout_features_ph: rollout_features,
            rollout_actions_ph: rollout_actions,
            rollout_rewards_ph: rollout_rewards
        }
        sess.run(rollout_update_ops, rollout_feed_dict)

        # for triple in zip(episode_features, episode_actions, episode_rewards):
        #     EXPERIENCE.append(triple)

    infeed_thread = threading.Thread(target=_run_infeed)
    outfeed_thread = threading.Thread(target=_run_outfeed)

    sess.run(tpu_init)
    sess.run(variables_init)
    sess.run(ds_inits)

    infeed_thread.start()
    outfeed_thread.start()

    for i in range(args.num_loop):
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
        default=10,
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
