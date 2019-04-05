#!/usr/bin/env bash

# Copyright 2018 Google Inc. All Rights Reserved.
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
#
# Convenience script for running ML training jobs.
#
# Prerequisites:
#   - Google Cloud SDK
#
# Globals:
#   PROJECT_ID: Google Cloud project to use.
#   BUCKET_ID: Google Cloud Storage bucket to store output.
#
# Arguments:
#  $1: Path or BigQuery table to dataset for ML training and eval
#  $2: Whether to run `local` (on-prem) or `remote` (GCP)


INPUT=$1
RUN_ENV=$2

if [[ ! "$RUN_ENV" =~ ^(local|remote)$ ]]; then
  RUN_ENV=local;
fi

NOW="$(date +"%Y%m%d_%H%M%S")"
PROJECT_NAME="sklearn_template"

JOB_NAME="${PROJECT_NAME}_training_${NOW}"
JOB_DIR="gs://$BUCKET_ID/$JOB_NAME"
PACKAGE_PATH=trainer
MAIN_TRAINER_MODULE=$PACKAGE_PATH.task

# TODO(cezequiel): Add support for hp tuning config
CONFIG_FILE=config.yaml

if [ "$RUN_ENV" = 'remote' ]; then
  RUN_ENV_ARGS="jobs submit training $JOB_NAME \
    --region us-central1
    --runtime-version 1.13
    --python-version 2.7
    --config $CONFIG_FILE \
    "
else  # assume `local`
  RUN_ENV_ARGS="local train"
fi

TRAINER_ARGS="\
  --input $INPUT \
  --log_level DEBUG \
  --num_samples 100 \
  "

CMD="gcloud ml-engine $RUN_ENV_ARGS \
  --job-dir $JOB_DIR \
  --module-name $MAIN_TRAINER_MODULE \
  --package-path $PACKAGE_PATH \
  -- \
  $TRAINER_ARGS \
  "

echo "Running command: $CMD"
eval "$CMD"
