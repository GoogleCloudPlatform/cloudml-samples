# TPU Templates

Templates for training TensorFlow models with Cloud TPUs.

## TPUEstimator

### Training on Cloud ML Engine

Run from the `templates` directory.

```
BUCKET="gs://YOUR-GCS-BUCKET/"

TRAINER_PACKAGE_PATH="tpu_estimator"
MAIN_TRAINER_MODULE="tpu_estimator.trainer"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="tpu_estimator_$now"

JOB_DIR=$BUCKET"tpu_estimator/"$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-central1 \
    --config config.yaml \
    --runtime-version 1.9 \
    -- \
    --model-dir=$JOB_DIR\
    --use-tpu
```

### Training on Compute Engine

TODO
