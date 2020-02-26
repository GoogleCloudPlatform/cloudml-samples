#!/bin/bash

REGION="choose-gcp-region" # choose a gcp region from https://cloud.google.com/ml-engine/docs/tensorflow/regions
BUCKET="your-bucket-name" # change to your bucket name

MODEL_NAME="you_model_name" # change to your estimator name
MODEL_VERSION="your_model_version" # change to your model version

MODEL_BINARIES=$(gsutil ls gs://${BUCKET}/models/${MODEL_NAME}/export/estimate | tail -1)

gsutil ls ${MODEL_BINARIES}

# delete model version
gcloud ai-platform versions delete ${MODEL_VERSION} --model=${MODEL_NAME} || true

# delete model
gcloud ai-platform models delete ${MODEL_NAME} || true

# deploy model to GCP
gcloud ai-platform models create ${MODEL_NAME} --regions=${REGION}

# deploy model version
gcloud ai-platform versions create ${MODEL_VERSION} --model=${MODEL_NAME} --origin=${MODEL_BINARIES} --runtime-version=1.13

# invoke deployed model to make prediction given new data instances
gcloud ai-platform predict --model=${MODEL_NAME} --version=${MODEL_VERSION} --json-instances=data/new-data.json
