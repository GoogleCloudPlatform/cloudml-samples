#!/bin/bash
# Copyright 2018 Google LLC
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


# default values
AGENT="TD3"
now=$(date +"%Y%m%d%H%M%S")
JOB_NAME="BipedalWalker_$now"
REGION="us-central1"
SCALE_TIER="BASIC_GPU"
MODULE_NAME="trainer.task"
PACKAGE_PATH="./trainer"
HP_CONFIG="./hptuning_config.yaml"
HP_TUNING=0

# parse command line args
PARAMS=""
while (( "$#" )); do
  case "$1" in
    -a|--agent)
      AGENT=$2
      shift 2
      ;;
    -p|--project-id)
      PROJECT_ID=$2
      shift 2
      ;;
    -r|--region)
      REGION=$2
      shift 2
      ;;
    -s|--scale-tier)
      SCALE_TIER=$2
      shift 2
      ;;
    -t|--tuning)
      HP_TUNING=1
      shift
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS --$1"
      shift
      ;;
  esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"

BUCKET_NAME=${PROJECT_ID}
LOG_DIR="gs://${BUCKET_NAME}/${JOB_NAME}"

# submit job
CMLE_FLAGS="--job-dir $LOG_DIR \
            --scale-tier $SCALE_TIER \
            --runtime-version 1.10 \
            --module-name $MODULE_NAME \
            --package-path $PACKAGE_PATH \
            --region $REGION \
           "
if (($HP_TUNING==1))
then
  CMLE_FLAGS=$CMLE_FLAGS"--config $HP_CONFIG "
fi
PKG_FLAGS="--agent=$AGENT"$PARAMS
ALL_FLAGS=$CMLE_FLAGS"-- "$PKG_FLAGS
echo $ALL_FLAGS
gcloud --project=${PROJECT_ID} ml-engine jobs submit training $JOB_NAME $ALL_FLAGS

