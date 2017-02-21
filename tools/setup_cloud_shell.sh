#!/bin/bash

# Copyright 2016 Google Inc. All Rights Reserved.
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

# Sets up the development environment for Cloud ML on Cloud Shell.

# Quit early if any command fails.
set -ex

# Some pip installed tools are added to ~/.local/bin, which should be
# added to the PATH.
if [[ ":${PATH}:" != *":${HOME}/.local/bin:"* ]]; then
  echo "Updating PATH"
  PATH=${HOME}/.local/bin:${PATH}
  echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
fi

# Install Python development packages.
pip install --user -U pillow
pip install --user -U numpy pandas scikit-learn pyyaml
# Install scipy separately so that pip does not get killed.
pip install --user -U scipy
# Install TensorFlow.
pip install --user -U https://storage.googleapis.com/tensorflow/linux/cpu/debian/jessie/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
# Install the Cloud ML SDK.
pip install --user -U https://storage.googleapis.com/cloud-ml/sdk/cloudml-0.1.9-alpha.tar.gz

# Download the Cloud ML samples.
mkdir -p ~/google-cloud-ml
cd ~/google-cloud-ml
if [ -d samples ]; then
  # Back up the previous samples.
  OLD_DIR="samples_$(date +%Y-%m-%d_%H-%M-%S)"
  mv samples "${OLD_DIR}"
fi
curl -L -o cloudml-samples.zip https://github.com/GoogleCloudPlatform/cloudml-samples/archive/master.zip
unzip cloudml-samples.zip
mv cloudml-samples-master/ samples/

# Everything completed successfully.
echo "Success! Your environment has the required tools and dependencies."
