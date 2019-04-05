#!/usr/bin/env bash

# ML training script

# Environmental variables
# PROJECT_ID
# BUCKET_ID

INPUT=$1

NOW="$(date +"%Y%m%d_%H%M%S")"
PROJECT_NAME="sklearn_template"

JOB_NAME="${PROJECT_NAME}_training_${NOW}"
JOB_DIR="gs://$BUCKET_ID/$JOB_NAME"
PACKAGE_PATH=trainer
MAIN_TRAINER_MODULE=$PACKAGE_PATH.task
REGION=us-central1
RUNTIME_VERSION=1.13
PYTHON_VERSION=2.7

TRAINER_ARGS="\
  --input $INPUT \
  --log_level DEBUG \
  "

RUN_ENV="local train"

CMD="gcloud ml-engine $RUN_ENV \
  --job-dir $JOB_DIR \
  --module-name $MAIN_TRAINER_MODULE \
  --package-path $PACKAGE_PATH \
  -- \
  $TRAINER_ARGS \
  "

echo "Running command: $CMD"
eval "$CMD"
