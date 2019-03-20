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

import copy


def reverse_obs(obs):
    """Given observation from BipedalWalker-v2, flip to duplicate.

    obs[:4]   - body observations
    obs[4:9]  - leg 1
    obs[9:14] - leg 2
    """
    mirror_obs = copy.deepcopy(obs)
    tmp = copy.deepcopy(mirror_obs[4:9])
    mirror_obs[4:9] = mirror_obs[9:14]
    mirror_obs[9:14] = tmp
    return mirror_obs


def reverse_act(action):
    """Given action from BipedalWalker-v2, flip to duplicate.

    action[:2] - leg 1
    action[2:] - leg 2
    """
    mirror_act = copy.deepcopy(action)
    tmp = copy.deepcopy(mirror_act[:2])
    mirror_act[:2] = mirror_act[2:]
    mirror_act[2:] = tmp
    return mirror_act

