#!/bin/bash
# Copyright 2018 Google Inc. All Rights Reserved.
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

PROJECT_ID=${1}
now=$(date +"%Y%m%d%H%M%S")
REGION="us-central1"
BUCKET_NAME=${PROJECT_ID}
JOB_NAME="tuning_BipedalWalker_$now"
LOG_DIR="gs://${BUCKET_NAME}/${JOB_NAME}"
SCALE_TIER=BASIC_GPU
MODULE_NAME=trainer.task
PACKAGE_PATH=./trainer

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $LOG_DIR \
  --runtime-version 1.10 \
  --config ./hptuning_config.yaml \
  --module-name $MODULE_NAME \
  --package-path $PACKAGE_PATH \
  --region $REGION \
  -- \
  --log_dir $LOG_DIR \
  --record_video=False \
  --max_episodes 3000 \
  --eval_interval 100 \
  --agent TD3
