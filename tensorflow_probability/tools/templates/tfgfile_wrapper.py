# Copyright 2019 Google LLC
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

from functools import wraps
import tensorflow as tf


def tfgfile_wrapper(f):
    # When an example writes to local disk, change it to write to GCS.
    @wraps(f)
    def wrapper(*args, **kwargs):
        fname = kwargs.pop('fname')
        with tf.gfile.GFile(fname, 'w') as fobj:
            kwargs['fname'] = fobj
            return_value = f(*args, **kwargs)
        return return_value

    return wrapper
