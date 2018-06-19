
now=$(date +"%Y%m%d_%H%M%S")
# GCS_BUCKET="gs://my-gcs-bucket"
BUCKET=$GCS_BUCKET

JOB_NAME="tpu_$now"
JOB_DIR=$BUCKET"/"$JOB_NAME

STAGING_BUCKET=$BUCKET
REGION=us-central1
DATA_DIR=gs://cloud-tpu-test-datasets/fake_imagenet
OUTPUT_PATH=$JOB_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket $STAGING_BUCKET \
    --runtime-version 1.8 \
    --module-name resnet.resnet_main \
    --package-path resnet/ \
    --region $REGION \
    --config config_resnet.yaml \
    -- \
    --data_dir=$DATA_DIR \
    --model_dir=$OUTPUT_PATH \
    --resnet_depth=101 \
    --train_steps=1024
