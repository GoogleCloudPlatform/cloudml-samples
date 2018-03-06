#!/bin/bash

echo "Submitting a Cloud ML Engine job..."

REGION="europe-west1"
TIER="BASIC" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
BUCKET="you-bucket-name" # change to your bucket name

MODEL_NAME="your_model_name" # change to your model name

PACKAGE_PATH=trainer # this can be a gcs location to a zipped and uploaded package
TRAIN_FILES=gs://${BUCKET}/path/to/data/train-data-*.csv
VALID_FILES=gs://${BUCKET}/path/to/data/valid-data-*.csv
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
        --package-path=${PACKAGE_PATH} # use --packages if gcs location \
        --config=config.yaml \
        -- \
        --train-files=${TRAIN_FILES} \
		--train-steps=10000 \
        --num-epochs=10 \
        --train-batch-size=200 \
        --eval-files=${VALID_FILES} \
        --eval-batch-size=200 \
        --learning-rate=0.01 \
        --hidden-units="64,32,10" \
        --layer-sizes-scale-factor=0.5 \
        --num-layers=3 \
        #--reuse-job-dir # use to resume training





