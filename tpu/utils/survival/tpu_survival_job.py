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

import sys
from time import sleep

from tpu_survival import TPUSurvival

project = 'PROJECT_ID'
location = 'LOCATION'
max_attempt = 10

ts = TPUSurvival(project=project, location=location)


while ts.current_index <= max_attempt:
    # creatre TPU pod
    ts.create()

    # wait until it is ready
    while ts.state != 'READY':
        sleep(10)
        ts.update_state()

    ts.run_task()

    while ts.state == 'READY':
        sleep(10)
        ts.update_state()

        # check the status of the training process.
        ts.current_process.poll()
        returncode = ts.current_process.returncode

        # training finished
        if returncode is not None:
            print('Training process terminated with code: {}.'.format(
                returncode))

            # clean up
            ts.delete()

            sys.exit(returncode)

    # when preempted
    ts.delete()
    ts.kill_current_task()

    # get ready for the next attempt
    ts.increment_index()
