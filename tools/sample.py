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

"""Multiline docstrings should
work but could be problematic.
"""

# This is safer.

"""Sample of what to_ipynb.py does"""

# Consecutive Comments are grouped into the same markdown cell.
# The leading '#' symbol is removed so the markdown cells look better.

# *It is okay to use [markdown](https://www.google.com/search?q=markdown).*

import argparse
import os

# Consecutive imports are grouped into a cell.
# Comments cause a new cell to be created, but blank lines between imports are ignored.

# This next import should say `from helpers import ...` even if its source says `from module.helpers import ...`
# Code manipulation is registered in `samples.yaml`.
from module.helpers import (
    some_function)

import yyy
import zzz

# Top level classes, function definitions, and expressions are in their own cells.
class A(object): # Inline comments are left as is.
    # Inner comments are left as is.
    def __init__(self):
        pass

class B(object):
    pass

def func(arg):
    """Docstrings are left as is"""
    def inner_func():
        print(arg)
    return inner_func

a = A()
print(a)

# This is a markdown cell.
def main(args):
    help(func)

# The last thing of the .py file must be the `if __name__ == '__main__':` block.
if __name__ == '__main__':
    # Its content is grouped into the last code cell.

    # All args should have a default value if the notebook is expected to be runnable without code change.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        help='Job dir',
        default='/tmp/sample'
    )

    # Use parse_known_args to ignore args passed in when running as a notebook.
    args, _ = parser.parse_known_args()

    main(args)
