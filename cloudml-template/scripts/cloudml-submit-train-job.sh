#!/bin/bash

echo "Submitting a Cloud ML Engine job..."

REGION="europe-west1"
TIER="BASIC" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
BUCKET="you-bucket-name" # change to your bucket name

MODEL_NAME="your_model_name" # change to your model name

PACKAGE_PATH=trainer # this can be a gcs location to a zipped and uploaded package
TRAIN_FILES=gs://${BUCKET}/path/to/data/train-data-*.csv
EVAL_FILES=gs://${BUCKET}/path/to/data/eval-data-*.csv
MODEL_DIR=gs://${BUCKET}/path/to/models/${MODEL_NAME}

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${TIER}_${CURRENT_DATE}
#JOB_NAME=tune_${MODEL_NAME}_${CURRENT_DATE} # for hyper-parameter tuning jobs

gcloud ml-engine jobs submit training ${JOB_NAME} \
        --job-dir=${MODEL_DIR} \
        --runtime-version=1.4 \
        --region=${REGION} \
        --scale-tier=${TIER} \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH}  \
        --config=config.yaml \
        -- \
        --train-files=${TRAIN_FILES} \
        --eval-files=${EVAL_FILES} \
	--train-steps=10000


# notes:
# use --packages instead of --package-path if gcs location
# add --reuse-job-dir to resume training
