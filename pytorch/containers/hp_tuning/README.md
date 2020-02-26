# Train a PyTorch Model on AI Platform using Custom Containers and Hyperparameter Tuning
This tutorial covers how to train a PyTorch model on AI Platform with Hyperparameter Tuning
using a Custom Container (docker image). The PyTorch model predicts whether the given sonar
signals are bouncing off a metal cylinder or off a cylindrical rock from
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29).

Citation: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and
Computer Science.

# How to use Hyperparameter Tuning to train a PyTorch model
1. Create your model
    1. Add argument parsing for the hyperparameter values. (These values are chosen for you in
    this tutorial)
    1. Add code to track the performance of your hyperparameter values.
1. Create the docker image
1. Build the docker image
1. Test your docker image locally
1. Deploy the docker image to Cloud Container Registry
1. Submit your training job

# Prerequisites
Before you jump in, let’s cover some of the different tools you’ll be using to get your container up and running on AI Platform.

[Google Cloud Platform](https://cloud.google.com/) lets you build and host applications and websites, store data, and analyze data on Google's scalable infrastructure.

[AI Platform](https://cloud.google.com/ml-engine/) is a managed service that enables you to easily build machine learning models that work on any type of data, of any size.

[Cloud Container Registry](https://cloud.google.com/container-registry/) is a single place for your team to manage Docker images, perform vulnerability analysis, and decide who can access what with fine-grained access control.

[Google Cloud Storage](https://cloud.google.com/storage/) (GCS) is a unified object storage for developers and enterprises, from live data serving to data analytics/ML to data archiving.

[Cloud SDK](https://cloud.google.com/sdk/) is a command line tool which allows you to interact with Google Cloud products. In order to run this tutorial, make sure that Cloud SDK is [installed](https://cloud.google.com/sdk/downloads) in the same environment as your Jupyter kernel.

[Overview of Hyperparameter Tuning](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview) - Hyperparameter tuning takes advantage of the processing infrastructure of Google Cloud Platform to test different hyperparameter configurations when training your model.

[docker](https://www.docker.com/) is a containerization technology that allows developers to package their applications and dependencies easily so that they can be run anywhere.

# Part 0: Setup
* [Create a project on GCP](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
* [Create a Google Cloud Storage Bucket](https://cloud.google.com/storage/docs/quickstart-console)
* [Enable AI Platform Training and Prediction, Container Registry, and Compute Engine APIs](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component,containerregistry.googleapis.com)
* [Install Cloud SDK](https://cloud.google.com/sdk/downloads)
* [Install docker](https://docs.docker.com/install/)
* [Configure docker for Cloud Container Registry](https://cloud.google.com/container-registry/docs/pushing-and-pulling)
* [Install PyTorch](https://pytorch.org/get-started/locally/) [Optional: used if running locally]
* [Install pandas](https://pandas.pydata.org/pandas-docs/stable/install.html) [Optional: used if running locally]

These variables will be needed for the following steps.

**Replace these variables:**
```
# PROJECT_ID: your project's id. Use the PROJECT_ID that matches your Google Cloud Platform project.
export PROJECT_ID=YOUR_PROJECT_ID

# BUCKET_ID: the bucket id you created above.
export BUCKET_ID=BUCKET_ID
```

Additional variables:
```
# JOB_DIR: with the path to a Google Cloud Storage location to use for job output.
export JOB_DIR=gs://$BUCKET_ID/hp_tuning

# IMAGE_REPO_NAME: where the image will be stored on Cloud Container Registry
export IMAGE_REPO_NAME=sonar_hp_tuning_pytorch_container

# IMAGE_TAG: an easily identifiable tag for your docker image
export IMAGE_TAG=sonar_hp_tuning_pytorch

# IMAGE_URI: the complete URI location for Cloud Container Registry
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
export REGION=us-central1

# JOB_NAME: the name of your job running on AI Platform.
export JOB_NAME=hp_tuning_container_job_$(date +%Y%m%d_%H%M%S)
```

# Part 1: Create the model you want to train
Here we provide an example [model.py](model.py) that trains a PyTorch model
to predict whether the given sonar signals are bouncing off a metal cylinder or off a cylindrical
rock. You will also find the code that handles argument parsing for the hyperparameter values as
well as the code to track the performance for hptuning.

Open up the [task.py](task.py) to see exactly how the model is called during
training.

[data_utils.py](data_utils.py) is used to download / load the data
and exports your trained model and uploads the model to Google Cloud Storage.

The dataset for the model is hosted originally at the UCI Machine Learning Repository. We've 
[hosted the sonar dataset in Cloud Storage](https://storage.cloud.google.com/cloud-samples-data/ml-engine/sonar/sonar.all-data?organizationId=433637338589&_ga=2.163217084.-1279615720.1534888758)
for use with this sample.

# Part 2: Create the docker Image
Open the [Dockerfile](Dockerfile) to see how the Docker image is created that will run on Cloud
AI Platform.

# Part 3: Build the docker Image
```
docker build -f Dockerfile -t $IMAGE_URI ./
```

# Part 4: Test your docker image locally
```
docker run $IMAGE_URI --epochs 1
```

If it ran successfully, the output should look similar to: `Accuracy: 58%`.

# Part 5: Deploy the docker image to Cloud Container Registry
You should have configured docker to use Cloud Container Registry, found
[here](https://cloud.google.com/container-registry/docs/pushing-and-pulling).
```
docker push $IMAGE_URI
```

# Part 6: Submit your training job
Open `hptuning_config.yaml` to see how to configure the hyper parameters that are passed into the
model.

Submit the training job to AI Platform using `gcloud`.

Note: You may need to install gcloud beta to submit the training job.
```
gcloud components install beta
```
```
gcloud beta ml-engine jobs submit training $JOB_NAME \
  --job-dir=$JOB_DIR \
  --region=$REGION \
  --master-image-uri $IMAGE_URI \
  --config=hptuning_config.yaml \
  --scale-tier BASIC
```

# [Optional] StackDriver Logging
You can view the logs for your training job:

1. Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
1. Select "Logging" in left-hand pane
1. Select "Cloud ML Job" resource from the drop-down
1. In filter by prefix, use the value of $JOB_NAME to view the logs

# [Optional] Verify Model File in GCS
View the contents of the destination model folder to verify that model file has indeed been
uploaded to GCS.

Note: The model can take a few minutes to train and show up in GCS.
```
gsutil ls gs://$BUCKET_ID/hp_tuning/*
```
