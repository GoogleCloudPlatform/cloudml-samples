#!/bin/bash

REGION="europe-west1"
BUCKET="your-bucket-name" # change to your bucket name

MODEL_NAME="you_model_name" # change to your estimator name
MODEL_VERSION="your.model.version" # change to your model version

MODEL_BINARIES=$(gsutil ls gs://${BUCKET}/models/${MODEL_NAME}/export/estimator | tail -1)

gsutil ls ${MODEL_BINARIES}

# delete model version
gcloud ml-engine versions delete ${MODEL_VERSION} --model=${MODEL_NAME}

# delete model
gcloud ml-engine models delete ${MODEL_NAME}

# deploy model to GCP
gcloud ml-engine models create ${MODEL_NAME} --regions=${REGION}

# deploy model version
gcloud ml-engine versions create ${MODEL_VERSION} --model=${MODEL_NAME} --origin=${MODEL_BINARIES} --runtime-version=1.4

# invoke deployed model to make prediction given new data instances
gcloud ml-engine predict --model=${MODEL_NAME} --version=${MODEL_VERSION} --json-instances=data/new-data.json