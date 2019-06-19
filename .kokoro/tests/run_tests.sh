#!/bin/bash
# Copyright 2019 Google LLC
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
# This is the common setup.
set -eo pipefail

export GCP_SDK_VERSION="247.0.0"
export PROJECT="cloud-devrel-kokoro-resources"
export REGION="us-central1"
export REQUIREMENTS_FILE="requirements.txt"
export KEYFILE="${KOKORO_GFILE_DIR}/keyfile.json"


check_if_changed(){
    # Ignore this test if there are no changes.
    cd ${KOKORO_ARTIFACTS_DIR}/github/cloudml-samples/${CMLE_TEST_DIR}
    # Check if a change happened to directory.
    DIFF=`git diff master $KOKORO_GITHUB_PULL_REQUEST_COMMIT $PWD`
    echo "GIT DIFF:\n $DIFF"
    if [[ -z $DIFF ]]; then
        echo "TEST IGNORED; directory not modified in pull request $KOKORO_GITHUB_PULL_REQUEST_NUMBER"
        exit 0
    fi
}


create_virtualenv(){
    sudo pip install virtualenv
    virtualenv ${KOKORO_ARTIFACTS_DIR}/envs/venv
    source ${KOKORO_ARTIFACTS_DIR}/envs/venv/bin/activate
    # Install dependencies.
    pip install --upgrade -r "${REQUIREMENTS_FILE}"
}


project_setup(){
    # Update SDK for gcloud ai-platform command.
    gcloud components update --version "${GCP_SDK_VERSION}" --quiet
    export GOOGLE_APPLICATION_CREDENTIALS="${KEYFILE}"
    gcloud auth activate-service-account --key-file "${KEYFILE}"
    gcloud config set project "${PROJECT}"
    gcloud config set compute/region "${REGION}"
    gcloud config list
}


main(){
    check_if_changed
    project_setup
    create_virtualenv
    cd ${KOKORO_ARTIFACTS_DIR}
    # Run specific test.
    bash "${CMLE_TEST_SCRIPT}"
}

main
