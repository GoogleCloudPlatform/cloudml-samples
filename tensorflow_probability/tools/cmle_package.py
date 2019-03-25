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

import os
import shutil
import urllib2

SOURCE_BASE = "https://raw.githubusercontent.com/tensorflow/probability/{branch}/tensorflow_probability/examples/{source_name}"
BRANCH = 'r0.5'


class CMLEPackage(object):
    def __init__(self, source_name, requires=None, tfgfile_wrap=None):
        self.source_name = source_name
        self.tfgfile_wrap = tfgfile_wrap

        self._source_content = None

        name = source_name.split('.')[0]
        self.output_dir = os.path.join('..', name)

        # prefix to output filename mapping.
        self.outputs = {
            '': ['setup.py', 'config.yaml', 'submit.sh'],
            'trainer': [self.source_name, '__init__.py', 'tfgfile_wrapper.py']
        }

        # clean up previously generated package
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, 'trainer'))

        self.format_dict = {
            'source_name': source_name,
            'name': name,
            'requires': '' if requires is None else ','.join("'{}'".format(req) for req in requires)
        }


    def get_and_modify_source(self):
        print('Getting source: {}'.format(self.source_name))
        source_url = SOURCE_BASE.format(branch=BRANCH, source_name=self.source_name)
        response = urllib2.urlopen(source_url)

        if self.tfgfile_wrap:
            lines = []
            add_import = True
            for line in response:
                if add_import and 'import' in line and 'from __future__' not in line:
                    line = 'from trainer.tfgfile_wrapper import tfgfile_wrapper\n' + line
                    add_import = False

                for to_wrap in self.tfgfile_wrap:
                    if 'def {}'.format(to_wrap) in line:
                        line = '@tfgfile_wrapper\n' + line

                lines.append(line)

            self._source_content = ''.join(lines)

        else:
            self._source_content = response.read()


    @property
    def source_content(self):
        if self._source_content is None:
            self.get_and_modify_source()
        return self._source_content


    def generate(self):
        for prefix, filenames in self.outputs.items():
            for filename in filenames:
                output_path = os.path.join(self.output_dir, prefix, filename)

                if filename == self.source_name:
                    content = self.source_content
                else:
                    with open(os.path.join('templates', filename), 'r') as f:
                        template = f.read()

                    content = template.format(**self.format_dict)

                with open(output_path, 'w') as f:
                    f.write(content)
