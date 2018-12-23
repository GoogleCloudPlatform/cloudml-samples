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

import numpy as np
import tensorflow as tf


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    """

    def __init__(self,
                 sess,
                 state_dim,
                 action_dim,
                 action_high,
                 action_low,
                 learning_rate,
                 grad_norm_clip,
                 tau,
                 batch_size,
                 name=None):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_high = action_high
        self.a_low = action_low
        self.learning_rate = learning_rate
        self.grad_norm_clip = grad_norm_clip
        self.tau = tau
        self.batch_size = batch_size

        # create networks
        net_name = 'actor' if name is None else name
        with tf.variable_scope(net_name):
            self.obs, self.action = self.create_actor_network()
        self.params = tf.trainable_variables(scope=net_name)
        with tf.variable_scope(net_name + '_target'):
            self.target_obs, self.target_action = self.create_actor_network()
        self.target_params = tf.trainable_variables(scope=net_name + '_target')

        # create ops
        (self.update_target_op,
         self.action_gradient,
         self.train_op) = self.create_actor_ops()

    def create_actor_ops(self):
        """Create training related ops."""
        target_update_op = tf.group([
            x.assign(tf.multiply(y, self.tau) + tf.multiply(x, 1. - self.tau))
            for x, y in zip(self.target_params, self.params)])
        grad_ph = tf.placeholder(tf.float32, [None, self.a_dim])
        actor_gradients = tf.gradients(self.action, self.params, -grad_ph)
        clipped_grad = map(
            lambda x: tf.clip_by_norm(tf.div(x, self.batch_size),
                                      self.grad_norm_clip), actor_gradients)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_grad, self.params))
        return target_update_op, grad_ph, train_op

    def create_actor_network(self):
        """A simple MLP."""
        s_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim])
        x = tf.layers.dense(inputs=s_inputs,
                            units=512,
                            activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x,
                            units=256,
                            activation=tf.nn.relu)
        actions = tf.layers.dense(inputs=x,
                                  units=self.a_dim,
                                  activation=tf.nn.sigmoid)
        scaled_actions = actions * (self.a_high - self.a_low) + self.a_low
        return s_inputs, scaled_actions

    def train(self, obs, a_gradient):
        self.sess.run(self.train_op,
                      feed_dict={self.obs: obs,
                                 self.action_gradient: a_gradient})

    def get_action(self, obs):
        if np.ndim(obs) == 1:
            obs = np.expand_dims(obs, axis=0)
        return self.sess.run(self.action, feed_dict={self.obs: obs})

    def get_target_action(self, obs):
        if np.ndim(obs) == 1:
            obs = np.expand_dims(obs, axis=0)
        return self.sess.run(self.target_action,
                             feed_dict={self.target_obs: obs})

    def update_target_network(self):
        self.sess.run(self.update_target_op)


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self,
                 sess,
                 state_dim,
                 action_dim,
                 learning_rate,
                 tau,
                 gamma,
                 name=None):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # build networks
        net_name = 'critic' if name is None else name
        with tf.variable_scope(net_name):
            (self.obs,
             self.action,
             self.q_value) = self.create_critic_network()
        self.params = tf.trainable_variables(scope=net_name)
        with tf.variable_scope(net_name + '_target'):
            (self.target_obs,
             self.target_action,
             self.target_q_value) = self.create_critic_network()
        self.target_params = tf.trainable_variables(scope=net_name + '_target')

        # build ops
        (self.update_target_op,
         self.y_ph,
         self.train_op,
         self.action_grad) = self.create_critic_ops()

    def create_critic_ops(self):
        """Create training related ops."""
        target_update_op = tf.group([
            x.assign(tf.multiply(y, self.tau) + tf.multiply(x, 1. - self.tau))
            for x, y in zip(self.target_params, self.params)])
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        loss = tf.reduce_mean(tf.losses.mean_squared_error(
            y_ph, self.q_value))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss, var_list=self.params)
        action_grads = tf.gradients(self.q_value, self.action)
        return target_update_op, y_ph, train_op, action_grads

    def create_critic_network(self):
        """A simple MLP."""
        s_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim])
        a_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.a_dim])
        inputs = tf.concat(values=[s_inputs, a_inputs], axis=1)
        x = tf.layers.dense(inputs=inputs,
                            units=512,
                            activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x,
                            units=256,
                            activation=tf.nn.relu)
        q_value = tf.squeeze(tf.layers.dense(inputs=x, units=1))
        return s_inputs, a_inputs, q_value

    def train(self, obs, action, y):
        return self.sess.run(self.train_op, feed_dict={
            self.obs: obs,
            self.action: action,
            self.y_ph: y
        })

    def get_qval(self, obs, action):
        if np.ndim(obs) == 1:
            obs = np.expand_dims(obs, axis=0)
        if np.ndim(action) == 1:
            action = np.expand_dims(action, axis=0)
        return self.sess.run(self.q_value, feed_dict={
            self.obs: obs,
            self.action: action
        })

    def get_target_qval(self, obs, action):
        if np.ndim(obs) == 1:
            obs = np.expand_dims(obs, axis=0)
        if np.ndim(action) == 1:
            action = np.expand_dims(action, axis=0)
        return self.sess.run(self.target_q_value, feed_dict={
            self.target_obs: obs,
            self.target_action: action
        })

    def get_action_gradients(self, obs, actions):
        return self.sess.run(self.action_grad, feed_dict={
            self.obs: obs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_op)

