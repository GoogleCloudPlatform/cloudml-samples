#!/bin/bash
set -e

### Ignore this test if there are no relevant changes
cd ${CMLE_REPO_DIR}/${CMLE_TEST_BASE_DIR}

if [ -z `git diff $KOKORO_GITHUB_PULL_REQUEST_COMMIT ${KOKORO_GITHUB_PULL_REQUEST_COMMIT}~1 $PWD` ]
then
    echo "TEST IGNORED; directory not modified in pull request $KOKORO_GITHUB_PULL_REQUEST_NUMBER"
    exit 0
fi

### Common setup


sudo apt-get install python-dev python-pip

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


sh $CMLE_TEST_SCRIPT
