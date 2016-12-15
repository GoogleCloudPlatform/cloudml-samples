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

# Sets up the development environment for Cloud ML on the Cloud Datalab Docker
# container.

# Quit early if any command fails.
set -ex

# Install the Cloud ML SDK.
pip install --upgrade --force-reinstall \
  https://storage.googleapis.com/cloud-ml/sdk/cloudml.latest.tar.gz

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
