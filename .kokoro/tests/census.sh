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
set -eo pipefail


download_files() {
    echo "Downloading files"
    # Copy files locally.
    CENSUS_DATA=gs://cloud-samples-data/ml-engine/census/data
    gsutil cp "${CENSUS_DATA}"/adult.data.csv census_data/adult.data.csv
    gsutil cp "${CENSUS_DATA}"/adult.test.csv census_data/adult.test.csv
}


run_tests() {
    # Run estimator tests.
    echo "Running '$1' code tests in `pwd`."
    # Change to directory
    cd $1
    # Download training and evaluation files
    download_files
    # Define AI Platform training
    MODEL_NAME="tensorflow_estimator"
    MODEL_DIR=trained_models/${MODEL_NAME}
    PACKAGE_PATH=trainer
    # Datasets
    TRAIN_FILES=census_data/adult.data.csv
    EVAL_FILES=census_data/adult.test.csv
    

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
    cd ${KOKORO_ARTIFACTS_DIR}/github/cloudml-samples/${CMLE_TEST_DIR}
    run_tests estimator
    echo 'Test was successful'
}

main
