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

from multiprocessing import Process
import os
import shlex
import signal
from subprocess import PIPE
from subprocess import Popen

from tpu_utils import create_tpu
from tpu_utils import delete_tpu
from tpu_utils import list_tpus

RUN_TASK_COMMAND = 'bash submit_preemptible.sh {tpu_name}'
TENSORFLOW_VERSION = '1.11'
ACCELERATOR_TYPE = 'v2-128'


class TPUSurvival(object):
    def __init__(self, project, location, prefix='tpu-node'):
        self.project = project
        self.location = location
        self.prefix = prefix

        # current running job
        self.current_index = 1
        self.current_process = None
        self.state = None

    def tpu_name(self, index=None):
        """Format tpu_name to be used in creation and deletion calls."""
        index = index or self.current_index
        return '{}-{}'.format(self.prefix, index)

    def tpu_cidr_block(self, index=None):
        """Format CIDR block to be used in creation calls."""
        index = index or self.current_index
        # This could overflow
        return '10.0.1{:0>2}.0/27'.format(str(index))

    def update_state(self):
        """Poll the TPU nodes and update self.state."""
        nodes = list_tpus(self.project, self.location).get('nodes', [])

        for node in nodes:
            name = node['name']
            tpu_name = name.split('/')[-1]
            health = node.get('health', None)
            state = node['state']

            print('TPU health/state: {}: {}/{}'.format(
                tpu_name, health, state))

            # The node that is running the current task.
            if tpu_name == self.tpu_name():
                self.state = state

    def kill_current_task(self):
        """Kill the current running task."""
        print('killing current process: {}'.format(self.current_index))

        # The subprocess runs a shell command, which in turn calls python.
        # This kills the whole process group with the shell command as the
        # process group leader.
        os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)

    def increment_index(self):
        self.current_index += 1

        print('current_index incremented to: {}'.format(self.current_index))

    # run_task should be called at the beginning and
    # then only after the call to kill current_process
    def run_task(self):
        """Call a subprocess to run the training task on the current TPU node.
        """
        tpu_name = self.tpu_name()

        print('running task: {}'.format(tpu_name))

        cmd = RUN_TASK_COMMAND.format(tpu_name=tpu_name)
        command = shlex.split(cmd)

        # use popen so we can kill it when needed
        p = Popen(command, stdout=PIPE, preexec_fn=os.setsid)

        self.current_process = p

    def delete(self, index=None):
        """Delete the TPU node of the given index.
        If the index is not provided the current node is deleted.
        """
        tpu_name = self.tpu_name(index)

        print('deleting: {}'.format(tpu_name))

        args = (self.project, self.location, tpu_name)
        p = Process(target=delete_tpu, args=args)
        p.start()

        return p

    def create(self, index=None):
        """Create a TPU node of the given index.
        If the index is not provided self.current_index is used.
        """
        tpu_name = self.tpu_name()
        tpu_cidr_block = self.tpu_cidr_block()

        print('creating: {}, {}'.format(tpu_name, tpu_cidr_block))

        args = (
            self.project, self.location, tpu_name, ACCELERATOR_TYPE,
            TENSORFLOW_VERSION, tpu_cidr_block, True
        )
        p = Process(target=create_tpu, args=args)
        p.start()

        return p
