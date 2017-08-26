#!/bin/bash

### Common setup
sudo apt-get install python-dev python-pip

echo Y | gcloud components update --version 167.0.0

curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

sudo pip install virtualenv
virtualenv ${KOKORO_ARTIFACTS_DIR}/envs/tf
source ${KOKORO_ARTIFACTS_DIR}/envs/tf/bin/activate

# Ubuntu stock distribution of pip is out of date
pip install --upgrade pip

# Install test requirements
pip install --upgrade -r $CMLE_REQUIREMENTS_FILE

gcloud auth activate-service-account --key-file "${KOKORO_GFILE_DIR}/${CMLE_KEYFILE}"
gcloud config set project $CMLE_PROJECT_ID
gcloud config set compute/region $CMLE_REGION

./$CMLE_TEST_SCRIPT
