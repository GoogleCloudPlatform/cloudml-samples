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

"""General interface of an RL agent.

The classes implements this class need to support the following interfaces:
1. random_action(observation), given an observation return a random action.
2. action(observation), given an observation return an action from the policy.
3. train(global_step), improves the agents internal policy once.

"""


class Agent(object):
    """General interface of an RL agent."""

    def initialize(self):
        """Initialization before playing.

        This function serves as a unified interface to do some pre-work.
        E.g., for DDPG and TD3, update target network should be done here.
        """
        pass

    def random_action(self, observation):
        """Return a random action.

        Given an observation return a random action.
        Specifications of the action space should be given/initialized
        when the agent is initialized.

        Args:
            observation: object, observations from the env.
        Returns:
            numpy.array, represent an action.
        """
        raise NotImplementedError('Not implemented')

    def action(self, observation):
        """Return an action according to the agent's internal policy.

        Given an observation return an action according to the agent's
        internal policy. Specifications of the action space should be
        given/initialized when the agent is initialized.

        Args:
            observation: object, observations from the env.
        Returns:
            numpy.array, represent an action.
        """
        raise NotImplementedError('Not implemented')

    def action_with_noise(self, observation):
        """Return a noisy action.

        Given an observation return a noisy action according to the agent's
        internal policy and exploration noise process.
        Specifications of the action space should be given/initialized
        when the agent is initialized.

        Args:
            observation: object, observations from the env.
        Returns:
            numpy.array, represent an action.
        """
        raise NotImplementedError('Not implemented')

    def train(self, global_step):
        """Improve the agent's policy once.

        Train the agent and improve its internal policy once.

        Args:
            global_step: int, global step count.
        Returns:
            object, represent training metrics.
        """
        raise NotImplementedError('Not implemented')
