
## Introduction

This sample shows how to run hyperparameter tuning jobs on Cloud Machine Learning Engine with Cloud TPUs using TensorFlow's `tf.metrics`.

This sample is adapted from [the official samples for training ResNet-50 with Cloud TPUs](https://github.com/tensorflow/tpu/tree/master/models/official/resnet) to run on Cloud Machine Learning Engine.


## Requirements

- Install [Google Cloud Platform SDK](https://cloud.google.com/sdk/).  The SDK includes the commandline tools `gcloud` for submitting training jobs to [Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/).

- Follow the steps [here](https://cloud.google.com/ml-engine/docs/tensorflow/using-tpus#authorize_your_tpu_name_short_to_access_your_project) to authorize Cloud TPU to access your project.

