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

# ! pip install tf_agents
# ! apt-get install python-box2d

import argparse
from functools import partial
import numpy as np
import os
import random
import time
import datetime

import tensorflow as tf
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

from tf_agents.environments import suite_gym
# from tf_agents.environments import tf_py_environment

import threading
from Queue import Queue


GYM_ENV = 'BipedalWalker-v2'
FEATURE_SIZE = 24
ACTION_SIZE = 4

# size of the experience gathered at each rollout phase
ROLLOUT_LENGTH = 1024

# the number of rollouts needed to fill up the experience cache
N_ROLLOUTS = 32
EXPERIENCE_LENGTH = ROLLOUT_LENGTH * N_ROLLOUTS


# TODO: change this to a Q-network
def policy(features):
    with tf.variable_scope('agent', reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(features, 40, activation=tf.nn.tanh)
        hidden = tf.layers.dense(hidden, 40, activation=tf.nn.tanh)
        actions = tf.layers.dense(hidden, ACTION_SIZE, activation=tf.nn.tanh)

    return actions


def fit_batch(features, actions, rewards):
    # features are observations
    # import ipdb; ipdb.set_trace()

    policy_actions = policy(features)
    loss = -tf.reduce_sum(rewards * tf.reduce_sum(tf.math.squared_difference(actions, policy_actions), axis=-1))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    return global_step, loss, train_op


def tpu_computation_with_infeed(batch_size, num_shards):
    # TPU device perspective.

    features, actions, rewards = tf.contrib.tpu.infeed_dequeue_tuple(
        # the dtypes and shapes need to be consistent with what is fed into the infeed queue.
        dtypes=[tf.float32, tf.float32, tf.float32],
        shapes=[
            (batch_size // num_shards, FEATURE_SIZE),
            (batch_size // num_shards, ACTION_SIZE),
            (batch_size // num_shards, )
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


# def make_ds(v, batch_size):
#     dataset = tf.data.Dataset.from_tensor_slices(v)
#     dataset = dataset.repeat().shuffle(batch_size).batch(batch_size)
#     iterator = dataset.make_initializable_iterator()
#     next_batch = iterator.get_next()

#     # TPU needs to know the tensor shapes at graph building time
#     n_dim = len(next_batch.shape)
#     merge_shape = [batch_size] + [None] * (n_dim - 1)
#     shape = next_batch.shape.merge_with(merge_shape) 
#     next_batch.set_shape(shape)

#     return next_batch, iterator.initializer


def set_shape(tensor, batch_size):
    n_dim = len(tensor.shape)
    merge_shape = [batch_size] + [None] * (n_dim - 1)
    shape = tensor.shape.merge_with(merge_shape) 
    tensor.set_shape(shape)


def main(args):
    # use variables to store experience
    features_var = tf.get_variable('features', dtype=tf.float32, shape=[EXPERIENCE_LENGTH, FEATURE_SIZE], trainable=False)
    actions_var = tf.get_variable('actions', dtype=tf.float32, shape=[EXPERIENCE_LENGTH, ACTION_SIZE], trainable=False)
    rewards_var = tf.get_variable('rewards', dtype=tf.float32, shape=[EXPERIENCE_LENGTH], trainable=False)

    # wrap the experience variables in a dict to shuffle them together
    experience = {'features': features_var, 'actions': actions_var, 'rewards': rewards_var}

    dataset = tf.data.Dataset.from_tensor_slices(experience)
    dataset = dataset.repeat().shuffle(768).batch(args.train_batch_size)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    for tensor in next_batch.values():
        set_shape(tensor, args.train_batch_size)

    features = next_batch['features']
    actions = next_batch['actions']
    rewards = next_batch['rewards']

    ds_init = iterator.initializer

    infeed_ops, outfeed_ops = cpu_setup_feed(features, actions, rewards, num_shards=8)

    # Wrap the tpu computation function to be run in a loop.
    def computation_loop():
        return tf.contrib.tpu.repeat(args.iterations_per_loop, partial(tpu_computation_with_infeed, batch_size=args.train_batch_size, num_shards=8))

    tpu_computation_loop = tf.contrib.tpu.batch_parallel(computation_loop, num_shards=8)

    # CPU policy used for interacting with the environment
    # Batch size of 1 for rollout against a single environment.
    features_ph = tf.placeholder(dtype=tf.float32, shape=(1, FEATURE_SIZE)) 
    rollout_actions = policy(features_ph)

    # from distutils.version import StrictVersion

    # # strip off `-rc?'
    # tf_version = tf.__version__.split('-')[0]

    # if StrictVersion(tf_version) >= StrictVersion('1.13'):
    #     rollout_actions = tf.squeeze(tf.random.categorical(logits=rollout_logits, num_samples=1))
    # else:
    #     rollout_actions = tf.squeeze(tf.random.multinomial(logits=rollout_logits, num_samples=1))

    features_var_ph = tf.placeholder(dtype=features_var.dtype, shape=[ROLLOUT_LENGTH, FEATURE_SIZE])
    actions_var_ph = tf.placeholder(dtype=actions_var.dtype, shape=[ROLLOUT_LENGTH, ACTION_SIZE])
    rewards_var_ph = tf.placeholder(dtype=rewards_var.dtype, shape=[ROLLOUT_LENGTH])

    new_f = tf.concat([features_var[ROLLOUT_LENGTH:], features_var_ph], axis=0)
    new_a = tf.concat([actions_var[ROLLOUT_LENGTH:], actions_var_ph], axis=0)
    new_r = tf.concat([rewards_var[ROLLOUT_LENGTH:], rewards_var_ph], axis=0)

    update_features_op = tf.assign(features_var, new_f)
    update_actions_op = tf.assign(actions_var, new_a)
    update_rewareds_op = tf.assign(rewards_var, new_r)


    # rollout_actions = tf.squeeze(tf.random.multinomial(logits=rollout_logits, num_samples=1))

    # utility ops
    tpu_init = tf.contrib.tpu.initialize_system()
    tpu_shutdown = tf.contrib.tpu.shutdown_system()
    variables_init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(args.model_dir)
    summary_writer.add_graph(tf.get_default_graph())

    # get the TPU resource's grpc url
    # Note: when running on CMLE, args.tpu should be left as None
    tpu_grpc_url = TPUClusterResolver(tpu=args.tpu).get_master()
    sess = tf.Session(tpu_grpc_url)

    # Use separate threads to run infeed and outfeed.
    def _run_infeed():
        for i in range(args.max_steps):
            time.sleep(2)
            sess.run(infeed_ops)

            if i % args.save_checkpoints_steps == 0:
                print('infeed {}'.format(i))

    def _run_infeed1(input_queue):
        thread = threading.currentThread()
        while thread.do_work:
            if not input_queue.empty():
                i = input_queue.get()

                if i % args.save_checkpoints_steps == 0:
                    print('infeed {}'.format(i))

                for _ in range(args.iterations_per_loop):
                    sess.run(infeed_ops)

                input_queue.task_done()
            else:
                time.sleep(1)


    def _run_outfeed():
        for i in range(args.max_steps):
            outfeed_data = sess.run(outfeed_ops)

            if i % args.save_checkpoints_steps == 0:
                print('outfeed {}'.format(i))
                print('data returned from outfeed: {}'.format(outfeed_data))


    def _run_tpu_computation(tpu_queue):
        thread = threading.currentThread()
        while thread.do_work:
            if not tpu_queue.empty():
                v = tpu_queue.get()
                sess.run(tpu_computation_loop)
                print('tpu computation: {}'.format(v))

                tpu_queue.task_done()
            else:
                time.sleep(1)

    def state_to_features(state):
        return state

    # initialize env
    env = suite_gym.load(GYM_ENV)

    # In the main thread, interact with th environment and collect data into the experience variables.
    def run_rollout(update_queue, on_policy=False):
        start_time = time.time()

        ts = env.reset()
        state = ts.observation
        reward = ts.reward
        done = ts.is_last()

        step_features = state_to_features(state)

        batch_features = []
        batch_actions = []
        batch_rewards = []

        # for debugging
        batch_logits = []

        # collect data up to the point when a point is scored
        while True:
            # Since the CPU and the TPU share the model variables, this is using the updated policy.
            # step_actions = sess.run(rollout_actions, {features_ph: step_features})

            if on_policy:
                step_actions = sess.run(rollout_actions, {features_ph: step_features.reshape((1, -1))}).squeeze()
            else:
                step_actions = np.clip(np.random.randn(4), -1, 1)

            ts = env.step(step_actions)
            state = ts.observation
            reward = ts.reward
            done = ts.is_last()

            batch_features.append(step_features)
            batch_actions.append(step_actions)
            batch_rewards.append(reward)

            if done:
                if len(batch_features) < ROLLOUT_LENGTH:
                    ts = env.reset()
                    state = ts.observation
                    step_features = state_to_features(state)
                else:
                    break

        print('>>>>>>> collected {} steps, {}'.format(len(batch_features), time.time() - start_time))

        update_queue.put((batch_features, batch_actions, batch_rewards))


    def run_update(update_queue):
        thread = threading.currentThread()
        while thread.do_work:
            if not update_queue.empty():
                start_time = time.time()

                batch_features, batch_actions, batch_rewards = update_queue.get()

                batch_features = np.array(batch_features).squeeze()
                batch_actions = np.array(batch_actions)
                batch_rewards = np.array(batch_rewards)

                sum_reward = batch_rewards.sum()
                print('>>>>>>> {}'.format(sum_reward))

                with open('reward.txt', 'a') as f:
                    f.write('{},{}\n'.format(len(batch_features),sum_reward))

                sess.run(
                    [
                        update_features_op,
                        update_actions_op,
                        update_rewareds_op
                    ],
                    {
                        features_var_ph: batch_features[-ROLLOUT_LENGTH:],
                        actions_var_ph: batch_actions[-ROLLOUT_LENGTH:],
                        rewards_var_ph: batch_rewards[-ROLLOUT_LENGTH:]
                    }
                )
                print('updated experience, {}'.format(time.time() - start_time))

                update_queue.task_done()

    tpu_queue = Queue(maxsize=0)
    input_queue = Queue(maxsize=0)
    update_queue = Queue(maxsize=0)


    infeed_thread = threading.Thread(target=_run_infeed1, args=(input_queue,))
    infeed_thread.do_work = True

    outfeed_thread = threading.Thread(target=_run_outfeed)

    tpu_thread = threading.Thread(target=_run_tpu_computation, args=(tpu_queue,))
    tpu_thread.do_work = True

    update_thread = threading.Thread(target=run_update, args=(update_queue,))
    update_thread.do_work = True

    sess.run(tpu_init)
    sess.run(variables_init)
    sess.run(ds_init)

    update_thread.start()

    # fill up the experience buffer
    for _ in range(N_ROLLOUTS):
        run_rollout(update_queue, on_policy=False)

    input_queue.put(-1)

    infeed_thread.start()
    outfeed_thread.start()
    tpu_thread.start()
    

    for i in range(args.num_loops):
        print('Iteration: {}'.format(i))

        tpu_queue.put(i)
        input_queue.put(i)

        # run_rollout adds to update_queue
        run_rollout(update_queue, on_policy=True)

        gs = sess.run(tf.train.get_or_create_global_step())

        # if i % args.save_checkpoints_steps == 0:
        if i % 500 == 0:
            print('saving checkpoint')
            saver.save(sess, os.path.join(args.model_dir, 'model.ckpt'), global_step=gs)

    tpu_thread.do_work = False
    infeed_thread.do_work = False
    update_thread.do_work = False

    # input_queue.join()
    # tpu_queue.join()
    infeed_thread.join()
    outfeed_thread.join()
    tpu_thread.join()
    update_thread.join()

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
        default=1,
        help='The number of iterations on TPU before switching to CPU.')
    parser.add_argument(
        '--num-loops',
        type=int,
        default=6000,
        help='The number of times switching to CPU.')
    parser.add_argument(
        '--save-checkpoints-steps',
        type=int,
        default=10,
        help='The number of training steps before saving each checkpoint.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=8192,
        help='The training batch size.  The training batch is divided evenly across the TPU cores.')
    parser.add_argument(
        '--tpu',
        default=None,
        help='The name or GRPC URL of the TPU node.  Leave it as `None` when training on CMLE.')

    args, _ = parser.parse_known_args()

    args.max_steps = args.iterations_per_loop * args.num_loops

    main(args)
