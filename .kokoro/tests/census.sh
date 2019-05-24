#!/bin/bash
# Copyright 2019 Google Inc.
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
set -eo pipefail


project_setup() {
    gcloud components update --version "${GCLOUD_SDK_VERSION}" --quiet
    export GOOGLE_APPLICATION_CREDENTIALS="${KOKORO_GFILE_DIR}/${CENSUS_TEST_BASE_DIR}"
    gcloud auth activate-service-account --key-file "${KOKORO_GFILE_DIR}/${CMLE_KEYFILE}"
    gcloud config set project $CMLE_PROJECT_ID
    gcloud config set compute/region $CMLE_REGION
    gcloud config list
}


download_files() {
    echo "Downloading files"
    # Copy files locally.
	gsutil cp gs://cloud-samples-data/ml-engine/census/data/adult.data.csv census_data/adult.data.csv
	gsutil cp gs://cloud-samples-data/ml-engine/census/data/adult.test.csv census_data/adult.test.csv
}


run_script_local() {
    # Run estimator tests.
	cd github/cloudml-samples/$1
	echo "Running '$1' code tests."
	# Install dependencies.
	pip install --upgrade -r requirements.txt
    download_files # Download training and evaluation files
    PACKAGE_PATH=trainer
    TRAIN_FILES=census_data/adult.data.csv
    EVAL_FILES=census_data/adult.test.csv
    MODEL_NAME="estimator"
    MODEL_DIR=trained_models/${MODEL_NAME}

    echo "Training local ML model"
    gcloud ai-platform local train \
            --module-name=trainer.task \
            --package-path=${PACKAGE_PATH} \
            -- \
            --train-files=${TRAIN_FILES} \
            --eval-files=${EVAL_FILES} \
            --train-steps 1000 \
            --eval-steps 100 \
            --job-dir=${MODEL_DIR}
}


main(){
    # Ignore this test if there are no relevant changes
    cd ${CMLE_REPO_DIR}/${CENSUS_TEST_BASE_DIR}
    DIFF=`git diff master $KOKORO_GITHUB_PULL_REQUEST_COMMIT $PWD`
    echo "DIFF:\n $DIFF"
    if [ -z  $DIFF ]
    then
        echo "TEST IGNORED; directory not modified in pull request $KOKORO_GITHUB_PULL_REQUEST_NUMBER"
        exit 0
    fi
    project_setup
    run_script_local estimator
}



main