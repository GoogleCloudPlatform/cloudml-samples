
## Introduction

This sample shows how to run training jobs on AI Platform with Cloud TPUs using TensorFlow's `tf.metrics`.

This sample is adapted from [the official samples for training ResNet-50 with Cloud TPUs](https://github.com/tensorflow/tpu/tree/master/models/official/resnet) to run on AI Platform.


## Requirements

- Install [Google Cloud Platform SDK](https://cloud.google.com/sdk/).  The SDK includes the commandline tools `gcloud` for submitting training jobs to [AI Platform](https://cloud.google.com/ml-engine/).

- Enable [Cloud Storage](https://cloud.google.com/storage).

- Follow the steps [here](https://cloud.google.com/ml-engine/docs/tensorflow/using-tpus#authorize_your_tpu_name_short_to_access_your_project) to authorize Cloud TPU to access your project.


## Steps

1. Clone the repository.

    ```
    git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git
    ```

1. If you do not already have a Cloud Storage bucket, create one to be used for the training job.

    ```
    gsutil mb gs://[YOUR_GCS_BUCKET]
    export GCS_BUCKET="gs://[YOUR_GCS_BUCKET]"
    ```

1. Run the sample.  The included script will train ResNet-50 for 1024 steps using a fake dataset.

    ```
    cd cloudml-samples/tpu/training/resnet
    bash submit_resnet.sh
    ```

