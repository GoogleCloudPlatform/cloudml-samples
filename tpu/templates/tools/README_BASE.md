# TPU Templates

Templates for training TensorFlow models with Cloud TPUs.

<TABLE>

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

1. Create TPU Node

    Do either of the following:

    * Go to console.cloud.google.com/compute/tpus to create a TPU node.

    * Use gcloud commandline tool:

    ```
    gcloud compute tpus create TPU-NAME \
    --zone=ZONE \
    --network=default \
    --range=10.0.5.0/29 \
    --accelerator-type=v2-8 \
    --version=1.9
    ```

1. Create GCE VM in the same `ZONE`

    ...

1. Connect to the GCE VM

    ...

1. Run from the GCE VM:

    ```
    pip install tensorflow==1.9.0

    git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git

    cd cloudml-samples/tpu/templates

    python -m tpu_estimator.trainer \
    --use-tpu \
    --tpu TPU-NAME \
    --model-dir gs://YOUR-GCS-BUCKET/
    ```

### Training on Colab

1. Create TPU Node

    Do either of the following:

    * Go to console.cloud.google.com/compute/tpus to create a TPU node.

    * Use gcloud commandline tool:

    ```
    gcloud compute tpus create TPU-NAME \
    --zone=ZONE \
    --network=default \
    --range=10.0.5.0/29 \
    --accelerator-type=v2-8 \
    --version=1.9
    ```

1. Create GCE VM in the same `ZONE`

    ...

1. Connect to the GCE VM

    ```
    gcloud compute ssh $INSTANCE_NAME -- -L 8080:localhost:8080
    ```

1. Go to Colab, connect to local runtime with port `8080`.

