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

"""Implementation of an agent with 2 critics and 2 actors.

This file implements an agent based on TD3.
TD3 has 2 critics to give more accurate estimation of Q-values,
however it doesn't have much choice when making a decision due to the factor
there is only 1 actor.

We add one more actor to the agent, and at each iteration
action = argmax(min(q1, q2))

"""
import agent
from common import replay_buffer
from common.actor_critic import ActorNetwork
from common.actor_critic import CriticNetwork
import numpy as np


class C2A2(agent.Agent):
    """C2A2 agent."""

    def __init__(self, env, sess, config):
        """Initialize members."""
        state_dim = env.observation_space.shape[0]
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low
        self.batch_size = config.batch_size
        self.warmup_size = config.warmup_size
        self.gamma = config.gamma
        self.sigma = config.sigma
        self.sigma_tilda = config.sigma_tilda
        self.noise_cap = config.c
        self.train_interval = config.d
        self.actor1 = ActorNetwork(sess=sess,
                                   state_dim=state_dim,
                                   action_dim=self.action_dim,
                                   action_high=self.action_high,
                                   action_low=self.action_low,
                                   learning_rate=config.actor_lr,
                                   grad_norm_clip=config.grad_norm_clip,
                                   tau=config.tau,
                                   batch_size=config.batch_size,
                                   name='actor1')
        self.actor2 = ActorNetwork(sess=sess,
                                   state_dim=state_dim,
                                   action_dim=self.action_dim,
                                   action_high=self.action_high,
                                   action_low=self.action_low,
                                   learning_rate=config.actor_lr,
                                   grad_norm_clip=config.grad_norm_clip,
                                   tau=config.tau,
                                   batch_size=config.batch_size,
                                   name='actor2')
        self.critic1 = CriticNetwork(sess=sess,
                                     state_dim=state_dim,
                                     action_dim=self.action_dim,
                                     learning_rate=config.critic_lr,
                                     tau=config.tau,
                                     gamma=config.gamma,
                                     name='critic1')
        self.critic2 = CriticNetwork(sess=sess,
                                     state_dim=state_dim,
                                     action_dim=self.action_dim,
                                     learning_rate=config.critic_lr,
                                     tau=config.tau,
                                     gamma=config.gamma,
                                     name='critic2')
        self.replay_buffer = replay_buffer.ReplayBuffer(
            buffer_size=config.buffer_size)

    def initialize(self):
        """Initialization before playing."""
        self.update_targets()

    def random_action(self, observation):
        """Return a random action."""
        return self.env.action_space.sample()

    def action(self, observation):
        """Return an action according to the agent's policy."""
        action1 = self.actor1.get_action(observation)
        action2 = self.actor2.get_action(observation)
        q_val1 = self.get_qval(observation, action1)
        q_val2 = self.get_qval(observation, action2)
        q_val1 = np.expand_dims(q_val1, axis=-1)
        q_val2 = np.expand_dims(q_val2, axis=-1)
        return np.where(q_val1 > q_val2, action1, action2)

    def target_action(self, observation):
        """Return an action according to the agent's policy."""
        target_action1 = self.actor1.get_target_action(observation)
        target_action2 = self.actor2.get_target_action(observation)
        target_q_val1 = self.get_target_qval(observation, target_action1)
        target_q_val2 = self.get_target_qval(observation, target_action2)
        target_q_val1 = np.expand_dims(target_q_val1, axis=-1)
        target_q_val2 = np.expand_dims(target_q_val2, axis=-1)
        return np.where(target_q_val1 > target_q_val2,
                        target_action1,
                        target_action2)

    def action_with_noise(self, observation):
        """Return a noisy action."""
        if self.replay_buffer.size > self.warmup_size:
            action = self.action(observation)
        else:
            action = self.random_action(observation)
        noise = np.clip(np.random.randn(self.action_dim) * self.sigma,
                        -self.noise_cap, self.noise_cap)
        action_with_noise = action + noise
        return (np.clip(action_with_noise, self.action_low, self.action_high),
                action, noise)

    def store_experience(self, s, a, r, t, s2):
        """Save experience to replay buffer."""
        self.replay_buffer.add(s, a, r, t, s2)

    def train(self, global_step):
        """Train the agent's policy for 1 iteration."""
        if self.replay_buffer.size > self.warmup_size:
            s0, a, r, t, s1 = self.replay_buffer.sample_batch(self.batch_size)
            epsilon = np.clip(np.random.randn(self.batch_size, self.action_dim),
                              -self.noise_cap, self.noise_cap)
            target_actions = self.target_action(s1) + epsilon
            target_actions = np.clip(target_actions,
                                     self.action_low,
                                     self.action_high)
            target_qval = self.get_target_qval(s1, target_actions)
            t = t.astype(dtype=int)
            y = r + self.gamma * target_qval * (1 - t)
            self.critic1.train(s0, a, y)
            self.critic2.train(s0, a, y)
            if global_step % self.train_interval == 0:
                actions = self.actor1.get_action(s0)
                grads = self.critic1.get_action_gradients(s0, actions)
                self.actor1.train(s0, grads[0])
                actions = self.actor2.get_action(s0)
                grads = self.critic1.get_action_gradients(s0, actions)
                self.actor2.train(s0, grads[0])
                self.update_targets()

    def update_targets(self):
        """Update all target networks."""
        self.actor1.update_target_network()
        self.actor2.update_target_network()
        self.critic1.update_target_network()
        self.critic2.update_target_network()

    def get_target_qval(self, observation, action):
        """Get target Q-val."""
        target_qval1 = self.critic1.get_target_qval(observation, action)
        target_qval2 = self.critic2.get_target_qval(observation, action)
        return np.minimum(target_qval1, target_qval2)

    def get_qval(self, observation, action):
        """Get Q-val."""
        qval1 = self.critic1.get_qval(observation, action)
        qval2 = self.critic2.get_qval(observation, action)
        return np.minimum(qval1, qval2)
