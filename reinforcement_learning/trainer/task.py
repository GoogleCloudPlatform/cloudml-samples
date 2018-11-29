# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
from c2a2_agent import C2A2
from ddpg_agent import DDPG
from td3_agent import TD3
import gym
from gym import wrappers
import json
import numpy as np
import os
import tensorflow as tf
from common import util


FLAGS = flags.FLAGS
flags.DEFINE_float('critic_lr', 2e-4, 'critic learning rate')
flags.DEFINE_float('actor_lr', 1e-4, 'actor learning rate')
flags.DEFINE_float('gamma', 0.99, 'reward discounting factor')
flags.DEFINE_float('tau', 0.001, 'target network update ratio')
flags.DEFINE_float('sigma', 0.1, 'exploration noise standard deviation')
flags.DEFINE_float('sigma_tilda', 0.05,
                   'noise standard deviation for smoothing regularization')
flags.DEFINE_float('c', 0.15, 'noise cap')
flags.DEFINE_float('grad_norm_clip', 5.0, 'maximum allowed gradient norm')
flags.DEFINE_integer('buffer_size', 1000000, 'replay buffer size')
flags.DEFINE_integer('d', 2, 'target update interval')
flags.DEFINE_integer('warmup_size', 10000, 'warm up buffer size')
flags.DEFINE_integer('batch_size', 32, 'mini-batch size')
flags.DEFINE_integer('rand_steps', 10,
                     'number of steps to use random actions in a new episode')
flags.DEFINE_integer('max_episodes', 3000,
                     'maximum number of episodes to train')
flags.DEFINE_integer('eval_interval', 100, 'interval to test')
flags.DEFINE_integer('max_to_keep', 5, 'number of model generations to save')
flags.DEFINE_string('agent', 'DDPG', 'type of agent, one of [DDPG|TD3]')
flags.DEFINE_string('logdir', './results', 'dir to save logs and videos')
flags.DEFINE_string('job-dir', './results', 'dir to save logs and videos')
flags.DEFINE_boolean('record_video', True, 'whether to record video when testing')


def build_summaries():
    """Training and evaluation summaries."""
    # train summaries
    episode_reward = tf.placeholder(dtype=tf.float32, shape=[])
    summary_vars = [episode_reward]
    with tf.name_scope('Training'):
        reward = tf.summary.scalar("Reward", episode_reward)
    summary_ops = tf.summary.merge([reward])
    # eval summary
    eval_episode_reward = tf.placeholder(dtype=tf.float32, shape=[])
    eval_summary_vars = [eval_episode_reward]
    with tf.name_scope('Evaluation'):
        eval_reward = tf.summary.scalar("EvalReward", eval_episode_reward)
    eval_summary_ops = tf.summary.merge([eval_reward])

    return summary_ops, summary_vars, eval_summary_ops, eval_summary_vars


def log_metrics(sess, writer, summary_ops, summary_vals, metrics, test=False):
    """Log metrics."""
    ep_cnt, ep_r, steps, actions, noises = metrics
    if test:
        tf.logging.info(
            '[TEST] Episode: {:d} | Reward: {:.2f} | AvgReward: {:.2f} | '
            'Steps: {:d}'.format(ep_cnt, ep_r, ep_r / steps, steps))
    else:
        aa = np.array(actions).mean(axis=0).squeeze()
        nn = np.array(noises).mean(axis=0).squeeze()
        tf.logging.info(
            '| Episode: {:d} | Reward: {:.2f} | AvgReward: {:.2f} | '
            'Steps: {:d} | AvgAction: {} | AvgNoise: {}'.format(
                ep_cnt, ep_r, ep_r / steps, steps, aa, nn))
    summary_str = sess.run(summary_ops, feed_dict={summary_vals[0]: ep_r})
    writer.add_summary(summary_str, ep_cnt)
    writer.flush()


def test(env, agent):
    """Test the trained agent"""
    tf.logging.info('Testing ...')
    s = env.reset()
    ep_reward = 0
    ep_steps = 0
    done = False

    while not done:
        if ep_steps < FLAGS.rand_steps:
            action = agent.random_action(s)
        else:
            action = agent.action(s)
        s2, r, done, info = env.step(action.squeeze().tolist())
        ep_reward += r
        ep_steps += 1
        s = s2
    return ep_reward, ep_steps


def train():
    """Train."""

    trial_id =  json.loads(
        os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', '')
    log_dir = os.path.join(FLAGS.logdir, trial_id, 'log')
    video_dir = os.path.join(FLAGS.logdir, trial_id, 'video')
    model_path = os.path.join(
        FLAGS.logdir, trial_id, 'model/{}.ckpt'.format(FLAGS.agent))

    env = gym.make('BipedalWalker-v2')
    if FLAGS.record_video:
      eval_interval = FLAGS.eval_interval
      env = wrappers.Monitor(
          env, video_dir,
          video_callable = lambda ep: (ep + 1 - (ep + 1) / eval_interval
                                       ) % eval_interval == 0)

    (summary_ops, summary_vars,
     eval_summary_ops, eval_summary_vars) = build_summaries()

    with tf.Session() as sess:

        if FLAGS.agent == 'DDPG':
            agent = DDPG(env, sess, FLAGS)
        elif FLAGS.agent == 'TD3':
            agent = TD3(env, sess, FLAGS)
        elif FLAGS.agent == 'C2A2':
            agent = C2A2(env, sess, FLAGS)
        else:
            raise ValueError('Unknown agent type {}'.format(FLAGS.agent))

        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        tf.logging.info('Start to train {} ...'.format(FLAGS.agent))
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        agent.initialize()
        global_step = 0
        for i in xrange(FLAGS.max_episodes):

            s = env.reset()
            ep_reward = 0
            ep_steps = 0
            noises = []
            actions = []
            done = False

            while not done:
                if ep_steps < FLAGS.rand_steps:
                    action = agent.random_action(s)
                else:
                    action, action_org, noise = agent.action_with_noise(s)
                    noises.append(noise)
                    actions.append(action_org)
                action = action.squeeze()

                s2, r, done, info = env.step(action.tolist())
                ep_reward += r
                ep_steps += 1
                global_step += 1
                agent.store_experience(s, action, r, done, s2)

                # mirror observations and actions
                flipped_s = util.reverse_obs(s)
                flipped_s2 = util.reverse_obs(s2)
                flipped_a = util.reverse_act(action)
                agent.store_experience(
                    flipped_s, flipped_a, r, done, flipped_s2)

                agent.train(global_step)
                s = s2

                if done:
                    ep_cnt = i + 1
                    log_metrics(sess,
                                writer,
                                summary_ops,
                                summary_vars,
                                metrics=(ep_cnt,
                                         ep_reward,
                                         ep_steps,
                                         actions,
                                         noises))
                    if ep_cnt % FLAGS.eval_interval == 0:
                        eval_ep_reward, eval_ep_steps = test(env, agent)
                        eval_ep_cnt = ep_cnt / FLAGS.eval_interval
                        log_metrics(sess,
                                    writer,
                                    eval_summary_ops,
                                    eval_summary_vars,
                                    metrics=(eval_ep_cnt,
                                             eval_ep_reward,
                                             eval_ep_steps,
                                             None,
                                             None),
                                    test=True)
                        ckpt_path = saver.save(sess,
                                               model_path,
                                               global_step=global_step)
                        tf.logging.info('Model saved to {}'.format(ckpt_path))
    env.close()


def main(argv):
    del argv
    for k, v in FLAGS.flag_values_dict().iteritems():
        tf.logging.info('{}: {}'.format(k, v))
    train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    app.run(main)
