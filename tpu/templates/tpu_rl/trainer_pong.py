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
from functools import partial
import numpy as np
import os
import random
import time
import threading
import datetime

import tensorflow as tf
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

from tf_agents.environments import suite_gym
# from tf_agents.environments import tf_py_environment

from Queue import Queue

# Using the first channel of state downsampled by a factor of 2 as features
FEATURE_SIZE = 80 * 80

ACTIONS = [0, 2, 3]

# size of the experience gathered at each rollout phase
ROLLOUT_LENGTH = 1024

# the number of rollouts needed to fill up the experience cache
N_ROLLOUTS = 64
EXPERIENCE_LENGTH = ROLLOUT_LENGTH * N_ROLLOUTS

# helper taken from: # https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r.tolist()


def policy(features):
    with tf.variable_scope('agent', reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(features, 200, activation=tf.nn.relu)
        logits = tf.layers.dense(hidden, len(ACTIONS))

    return logits


def fit_batch(features, actions, rewards):
    # features are observations

    logits = policy(features)
    onehot_labels = tf.one_hot(actions, depth=len(ACTIONS))
    loss = tf.reduce_sum(rewards * tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
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
            (batch_size // num_shards, ),
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


def make_ds(v, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(v)
    dataset = dataset.repeat().shuffle(ROLLOUT_LENGTH).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    n_dim = len(next_batch.shape)
    merge_shape = [batch_size] + [None] * (n_dim - 1)
    shape = next_batch.shape.merge_with(merge_shape) 
    next_batch.set_shape(shape)

    return next_batch, iterator.initializer


def set_shape(tensor, batch_size):
    n_dim = len(tensor.shape)
    merge_shape = [batch_size] + [None] * (n_dim - 1)
    shape = tensor.shape.merge_with(merge_shape) 
    tensor.set_shape(shape)


def main(args):
    # use variables to store experience
    features_var = tf.get_variable('features', dtype=tf.float32, shape=[EXPERIENCE_LENGTH, FEATURE_SIZE], trainable=False)
    actions_var = tf.get_variable('actions', dtype=tf.int32, shape=[EXPERIENCE_LENGTH], trainable=False)
    rewards_var = tf.get_variable('rewards', dtype=tf.float32, shape=[EXPERIENCE_LENGTH], trainable=False)

    # wrap the experience variables in a dict to shuffle them together
    experience = {'features': features_var, 'actions': actions_var, 'rewards': rewards_var}

    dataset = tf.data.Dataset.from_tensor_slices(experience)
    dataset = dataset.repeat().shuffle(32).batch(args.train_batch_size)
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
    rollout_logits = policy(features_ph)

    from distutils.version import StrictVersion

    # strip off `-rc?'
    tf_version = tf.__version__.split('-')[0]

    if StrictVersion(tf_version) >= StrictVersion('1.13'):
        rollout_actions = tf.squeeze(tf.random.categorical(logits=rollout_logits, num_samples=1))
    else:
        rollout_actions = tf.squeeze(tf.random.multinomial(logits=rollout_logits, num_samples=1))

    # placeholders and ops for updating after rollout
    # features_var_ph = tf.placeholder(dtype=features_var.dtype, shape=features_var.shape)
    # actions_var_ph = tf.placeholder(dtype=actions_var.dtype, shape=actions_var.shape)
    # rewards_var_ph = tf.placeholder(dtype=rewards_var.dtype, shape=rewards_var.shape)
    features_var_ph = tf.placeholder(dtype=features_var.dtype, shape=[ROLLOUT_LENGTH, FEATURE_SIZE])
    actions_var_ph = tf.placeholder(dtype=actions_var.dtype, shape=[ROLLOUT_LENGTH])
    rewards_var_ph = tf.placeholder(dtype=rewards_var.dtype, shape=[ROLLOUT_LENGTH])


    # # TODO: replace a random segment of experience with sliced assign
    # # NOTE: this isn't really uniformly selecting the slice
    # # replace_start = tf.random.uniform(shape=(), minval=0, maxval=EXPERIENCE_LENGTH-ROLLOUT_LENGTH, dtype=tf.float32)
    # replace_start = 0
    # replace_end = replace_start + ROLLOUT_LENGTH

    # update_features_op = tf.assign(features_var[replace_start:replace_end], features_var_ph)
    # update_actions_op = tf.assign(actions_var[replace_start:replace_end], actions_var_ph)
    # update_rewareds_op = tf.assign(rewards_var[replace_start:replace_end], rewards_var_ph)


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

    # with tf.name_scope('summaries'):
    #     summary_reward = tf.placeholder(
    #         shape=(),
    #         dtype=tf.float32
    #     )

    #     # the weights to the hidden layer can be visualized
    #     hidden_weights = tf.trainable_variables()[0]
    #     for h in range(200):
    #         slice_ = tf.slice(hidden_weights, [0, h], [-1, 1])
    #         image = tf.reshape(slice_, [1, 105, 80, 1])
    #         tf.summary.image('hidden_{:04d}'.format(h), image)

    #     for var in tf.trainable_variables():
    #         tf.summary.histogram(var.op.name, var)
    #         tf.summary.scalar('{}_max'.format(var.op.name), tf.reduce_max(var))
    #         tf.summary.scalar('{}_min'.format(var.op.name), tf.reduce_min(var))
            
    #     tf.summary.scalar('rollout_reward', summary_reward)
    #     # tf.summary.scalar('loss', loss)

    #     merged = tf.summary.merge_all()

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
            if input_queue.empty():
                time.sleep(1)
            else:
                i = input_queue.get()

                if i % args.save_checkpoints_steps == 0:
                    print('infeed {}'.format(i))

                for _ in range(args.iterations_per_loop):
                    sess.run(infeed_ops)

                input_queue.task_done()


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

    # https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    def state_to_features(state):
        I = state[35:195]
        I = I[::2, ::2, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(float).reshape((1, FEATURE_SIZE))

    def action_to_env_action(action):
        if action in range(3):
            return ACTIONS[action]
        else:
            return random.choice(ACTIONS)

    # initialize env
    env = suite_gym.load('Pong-v0')

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
        while reward == 0 or len(batch_features) <= ROLLOUT_LENGTH:
            # Since the CPU and the TPU share the model variables, this is using the updated policy.
            # step_actions = sess.run(rollout_actions, {features_ph: step_features})

            # for debugging
            if on_policy:
                [step_actions, step_logits] = sess.run([rollout_actions, rollout_logits], {features_ph: step_features})
                # for debugging
                batch_logits.append(step_logits)
            else:
                step_actions = random.randint(0, 2)

            env_action = action_to_env_action(step_actions)

            ts = env.step(env_action)
            state = ts.observation
            reward = ts.reward
            done = ts.is_last()

            batch_features.append(step_features)
            batch_actions.append(step_actions)
            batch_rewards.append(reward)

            if done:
                ts = env.reset()
                state = ts.observation

            step_features = state_to_features(state)

        print('>>>>>>> collected {} steps, {}'.format(len(batch_features), time.time() - start_time))

        udpate_queue.put((batch_features, batch_actions, batch_rewards))


        # TODO: split the updating step into a separate thread

    def run_update(update_queue):
        while thread.do_work:
            if not update_queue.empty():
                start_time = time.time()

                batch_features, batch_actions, batch_rewards = update_queue.get()

                batch_features = np.array(batch_features).squeeze()
                batch_actions = np.array(batch_actions)
                batch_rewards = np.array(batch_rewards)

                # for debugging:
                # print(batch_actions)
                # print(batch_logits)
                sum_reward = batch_rewards.sum()
                print('>>>>>>> {}'.format(sum_reward))

                # summary, gs = sess.run([merged, tf.train.get_or_create_global_step()], feed_dict={summary_reward: sum_reward})
                # summary_writer.add_summary(summary, gs)

                # process the rewards
                batch_rewards = discount_rewards(batch_rewards, 0.95)
                batch_rewards -= np.mean(batch_rewards)
                batch_rewards /= np.std(batch_rewards)

                # fv, av, rv = sess.run([features_var, actions_var, rewards_var])
                # new_fv = np.concatenate([fv[ROLLOUT_LENGTH:], batch_features[-ROLLOUT_LENGTH:]])
                # new_av = np.concatenate([av[ROLLOUT_LENGTH:], batch_actions[-ROLLOUT_LENGTH:]])
                # new_rv = np.concatenate([rv[ROLLOUT_LENGTH:], batch_rewards[-ROLLOUT_LENGTH:]])

                # sess.run([update_features_op, update_actions_op, update_rewareds_op], {features_var_ph: new_fv, actions_var_ph: new_av, rewards_var_ph: new_rv})

                sess.run([update_features_op, update_actions_op, update_rewareds_op], {features_var_ph: batch_features[-ROLLOUT_LENGTH:], actions_var_ph: batch_actions[-ROLLOUT_LENGTH:], rewards_var_ph: batch_rewards[-ROLLOUT_LENGTH:]})
                print('updated experience, {}'.format(time.time() - start_time))

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

        if i % args.save_checkpoints_steps == 0:
            saver.save(sess, os.path.join(args.model_dir, 'model.ckpt'), global_step=gs)

    tpu_thread.do_work = False
    infeed_thread.do_work = False
    update_thread.do_work = False

    # input_queue.join()
    # tpu_queue.join()
    infeed_thread.join()
    outfeed_thread.join()
    tpu_thread.join()
    udpate_thread.join()

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
        default=100,
        help='The number of times switching to CPU.')
    parser.add_argument(
        '--save-checkpoints-steps',
        type=int,
        default=10,
        help='The number of training steps before saving each checkpoint.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=16384,
        help='The training batch size.  The training batch is divided evenly across the TPU cores.')
    parser.add_argument(
        '--tpu',
        default=None,
        help='The name or GRPC URL of the TPU node.  Leave it as `None` when training on CMLE.')

    args, _ = parser.parse_known_args()

    args.max_steps = args.iterations_per_loop * args.num_loops

    main(args)
