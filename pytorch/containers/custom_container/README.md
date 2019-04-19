# Train a PyTorch Model on AI Platform using a Custom Container
This tutorial covers how to create a Custom Container (docker image) to train a PyTorch model on
AI Platform. The PyTorch model predicts whether the given sonar signals are bouncing off a
metal cylinder or off a cylindrical rock from
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29).

Citation: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and
Computer Science.

# How to build a custom container to train a PyTorch model
1. Create your model
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

[Cloud SDK](https://cloud.google.com/sdk/) is a command line tool which allows you to interact with Google Cloud products. In order to run this notebook, make sure that Cloud SDK is [installed](https://cloud.google.com/sdk/downloads) in the same environment as your Jupyter kernel.

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
# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
export IMAGE_REPO_NAME=sonar_pytorch_container

# IMAGE_TAG: an easily identifiable tag for your docker image
export IMAGE_TAG=sonar_pytorch

# IMAGE_URI: the complete URI location for Cloud Container Registry
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
export REGION=us-central1

# JOB_NAME: the name of your job running on AI Platform.
export JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)
```

# Part 1: Create the model you want to train
Here we provide an example [model.py](model.py) that trains a PyTorch model
to predict whether the given sonar signals are bouncing off a metal cylinder or off a cylindrical
rock.

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

If it runs successfully, the output should be similar to:
```
[1,     6] loss: 0.707
[1,    12] loss: 0.685
[1,    18] loss: 0.693
[1,    24] loss: 0.706
[1,    30] loss: 0.685
[1,    36] loss: 0.690
[1,    42] loss: 0.696
[1,    48] loss: 0.688

Test set:
    Average loss: 2.8893
    Accuracy: 121/208 (58%)
```

# Part 5: Deploy the docker image to Cloud Container Registry
You should have configured docker to use Cloud Container Registry, found
[here](https://cloud.google.com/container-registry/docs/pushing-and-pulling).
```
docker push $IMAGE_URI
```

# Part 6: Submit your training job
Submit the training job to AI Platform using `gcloud`.

Note: You may need to install gcloud beta to submit the training job.
```
gcloud components install beta
```
```
gcloud beta ml-engine jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  --scale-tier BASIC \
  -- \
  --model-dir=$BUCKET_ID \
  --epochs=10
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
gsutil ls gs://$BUCKET_ID/sonar_*
```
