# Copyright 2018 Google LLC
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

import argparse
from c2a2_agent import C2A2
from ddpg_agent import DDPG
from td3_agent import TD3
import gym
from gym import wrappers
import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from common import util


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


def test(env, agent, config):
    """Test the trained agent"""
    tf.logging.info('Testing ...')
    s = env.reset()
    ep_reward = 0
    ep_steps = 0
    done = False

    while not done:
        if ep_steps < config.rand_steps:
            action = agent.random_action(s)
        else:
            action = agent.action(s)
        s2, r, done, info = env.step(action.squeeze().tolist())
        ep_reward += r
        ep_steps += 1
        s = s2
    return ep_reward, ep_steps


def train(config):
    """Train."""

    log_dir = os.path.join(config.job_dir, 'log')
    video_dir = os.path.join(config.job_dir, 'video')
    model_path = os.path.join(
        config.job_dir, 'model/{}.ckpt'.format(config.agent))

    env = gym.make('BipedalWalker-v2')
    if config.record_video:
      eval_interval = config.eval_interval
      env = wrappers.Monitor(
          env, video_dir,
          video_callable = lambda ep: (ep + 1 - (ep + 1) / eval_interval
                                       ) % eval_interval == 0)

    (summary_ops, summary_vars,
     eval_summary_ops, eval_summary_vars) = build_summaries()

    with tf.Session() as sess:

        if config.agent == 'DDPG':
            agent = DDPG(env, sess, config)
        elif config.agent == 'TD3':
            agent = TD3(env, sess, config)
        elif config.agent == 'C2A2':
            agent = C2A2(env, sess, config)
        else:
            raise ValueError('Unknown agent type {}'.format(config.agent))

        saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        tf.logging.info('Start to train {} ...'.format(config.agent))
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        agent.initialize()
        global_step = 0
        for i in xrange(config.max_episodes):

            s = env.reset()
            ep_reward = 0
            ep_steps = 0
            noises = []
            actions = []
            done = False

            while not done:
                if ep_steps < config.rand_steps:
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
                    if ep_cnt % config.eval_interval == 0:
                        eval_ep_reward, eval_ep_steps = test(env, agent, config)
                        eval_ep_cnt = ep_cnt / config.eval_interval
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


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--critic-lr',
      help='critic learning rate',
      type=float,
      default=2e-4)
  parser.add_argument(
      '--actor-lr',
      help='actor learning rate',
      type=float,
      default=1e-4)
  parser.add_argument(
      '--gamma',
      help='reward discounting factor',
      type=float,
      default=0.99)
  parser.add_argument(
      '--tau',
      help='target network update ratio',
      type=float,
      default=0.001)
  parser.add_argument(
      '--sigma',
      help='exploration noise standard deviation',
      type=float,
      default=0.1)
  parser.add_argument(
      '--sigma-tilda',
      help='noise standard deviation for smoothing regularization',
      type=float,
      default=0.05)
  parser.add_argument(
      '--c',
      help='noise cap',
      type=float,
      default=0.15)
  parser.add_argument(
      '--grad-norm-clip',
      help='maximum allowed gradient norm',
      type=float,
      default=5.0)
  parser.add_argument(
      '--buffer-size',
      help='replay buffer size',
      type=int,
      default=1000000)
  parser.add_argument(
      '--d',
      help='target update interval',
      type=int,
      default=2)
  parser.add_argument(
      '--warmup-size',
      help='warm up buffer size',
      type=int,
      default=10000)
  parser.add_argument(
      '--batch-size',
      help='mini-batch size',
      type=int,
      default=32)
  parser.add_argument(
      '--rand-steps',
      help='number of steps to user random actions in a new episode',
      type=int,
      default=10)
  parser.add_argument(
      '--max-episodes',
      help='maximum number of episodes to train',
      type=int,
      default=3000)
  parser.add_argument(
      '--eval-interval',
      help='interval to test',
      type=int,
      default=100)
  parser.add_argument(
      '--max-to-keep',
      help='number of model generations to keep',
      type=int,
      default=5)
  parser.add_argument(
      '--agent',
      help='type of agent, one of [DDPG|TD3|C2A2]',
      default='DDPG')
  parser.add_argument(
      '--job-dir',
      help='dir to save logs and videos',
      default='./results')
  parser.add_argument(
      '--record-video',
      help='whether to record video when testing',
      action='store_true')
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO')

  args, _ = parser.parse_known_args()
  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  for k, v in args.__dict__.iteritems():
    tf.logging.info('{}: {}'.format(k, v))

  config = hparam.HParams(**args.__dict__)
  train(config)
