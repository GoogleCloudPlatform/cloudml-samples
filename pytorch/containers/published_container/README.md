# Train a PyTorch Model on Cloud ML Engine using a Published Container
This tutorial covers how to use a Published Container (docker image) to train a PyTorch
model on Cloud ML Engine. In this way, you or your team can reuse your docker images for training
jobs. In this tutorial, you will use a publicly published docker image from Google to train a
model. The PyTorch model predicts whether the given sonar signals are bouncing off a metal
cylinder or off a cylindrical rock from
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29).

Citation: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and
Computer Science.

# How to use a published container to train a PyTorch model
1. Create your model
1. Test your trainer application locally and package it
1. Upload the package to Cloud Storage
1. Submit your training job

# Prerequisites
Before you jump in, let’s cover some of the different tools you’ll be using to get your container
up and running on ML Engine.

[Google Cloud Platform](https://cloud.google.com/) lets you build and host applications and
websites, store data, and analyze data on Google's scalable infrastructure.

[Cloud ML Engine](https://cloud.google.com/ml-engine/) is a managed service that enables you to
easily build machine learning models that work on any type of data, of any size.

[Cloud Container Registry](https://cloud.google.com/container-registry/) is a single place for
your team to manage Docker images, perform vulnerability analysis, and decide who can access what
with fine-grained access control.

[Google Cloud Storage](https://cloud.google.com/storage/) (GCS) is a unified object storage for
developers and enterprises, from live data serving to data analytics/ML to data archiving.

[Cloud SDK](https://cloud.google.com/sdk/) is a command line tool which allows you to interact
with Google Cloud products. In order to run this tutorial, make sure that Cloud SDK is
[installed](https://cloud.google.com/sdk/downloads) in the same environment as your Jupyter kernel.

[docker](https://www.docker.com/) is a containerization technology that allows developers to
package their applications and dependencies easily so that they can be run anywhere.

# Part 0: Setup
* [Create a project on GCP](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
* [Create a Google Cloud Storage Bucket](https://cloud.google.com/storage/docs/quickstart-console)
* [Enable Cloud Machine Learning Engine, Container Registry, and Compute Engine APIs](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component,containerregistry.googleapis.com)
* [Install Cloud SDK](https://cloud.google.com/sdk/downloads)
* [Install docker](https://docs.docker.com/install/)
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
# PUBLISHED_IMAGE_URI: the complete URI location for Cloud Container Registry
export PUBLISHED_IMAGE_URI=gcr.io/cloud-ml-public/pytorch_cpu_2_7:1_0_preview

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
export REGION=us-central1

# JOB_NAME: the name of your job running on Cloud ML Engine.
export JOB_NAME=published_container_job_$(date +%Y%m%d_%H%M%S)

# PACKAGE_PATH: A packaged training application that will be staged in a Google Cloud Storage
# location. The model file created below is placed inside this package path.
export PACKAGE_PATH=./sonar_training

# PACKAGE_NAME: The name of your packaged training application.
export PACKAGE_NAME=sonar_package.tar.gz

# TRAINER_MODULE: Tells ML Engine which file to execute. This is formatted as follows
# <folder_name.python_file_name>
export TRAINER_MODULE=trainer.task
```

# Part 1: Create the model you want to train
Here we provide an example [model.py](sonar_trainer/trainer/model.py) that trains a PyTorch model
to predict whether the given sonar signals are bouncing off a metal cylinder or off a cylindrical
rock.

Open up the [task.py](sonar_trainer/trainer/task.py) to see exactly how the model is called during
training.

[data_utils.py](sonar_trainer/trainer/data_utils.py) is used to download / load the data
and exports your trained model and uploads the model to Google Cloud Storage.

The dataset for the model is hosted originally at the UCI Machine Learning Repository. We've 
[hosted the sonar dataset in Cloud Storage](https://storage.cloud.google.com/cloud-samples-data/ml-engine/sonar/sonar.all-data?organizationId=433637338589&_ga=2.163217084.-1279615720.1534888758)
for use with this sample.

Note: In the next step, we will package the training application before testing it locally.

You'll notice on lines 20-21 of [task.py](sonar_training/trainer/data_utils.py) that the code
imports from our file structure.
```
from trainer import data_utils
from trainer import model
```

Further below we will package this application to run on ML Engine with the published docker image.

# Part 2: Package your trainer application and test it locally
To test our code locally, we first have to package our training application. As a brief overview,
the file structure is as follows:
```
- sonar_training
 - trainer
   - __init__.py
   - data_utils.py
   - model.py
   - task.py
 - setup.py
```

[setup.py](sonar_training/setup.py) - Installs the required dependencies for our package that are
not provided by our docker image. PyTorch is included in the image, but pandas and
google-cloud-storage are not.

[\_\_init__.py"](sonar_training/trainer/__init__.py) - Allows our task.py file to import the
data_utils.py and model.py files.

For more info about packaging a trainer application check out the docs
[here](https://cloud.google.com/ml-engine/docs/tensorflow/packaging-trainer)

Since the code is already structured correctly for packaging, we'll simply package it.
```
tar -zcvf $PACKAGE_NAME sonar_training/
```

Now we can test the packaged trainer application locally before using the published docker image.
This is usually a good practice to make sure your code works, before running a long training job on
Cloud ML Engine.

To do that, we will first install the package locally. It may be best to use a
[virtual environment](https://cloud.google.com/python/setup#installing_and_using_virtualenv)
so as not to modify your python workspace.

Note: If using a virtual environment, remove the `--user` flag.
```
pip install --user --upgrade --force-reinstall --no-deps $PACKAGE_NAME
```

Then run the package for 1 epoch to verify that it works.
```
python -m $TRAINER_MODULE --epochs=1
```

Lastly, as a bit of cleanup, we'll remove the locally installed package.
```
pip uninstall sonar_training -y
```

# Part 3: Upload the package to Cloud Storage
Now that the code has been packaged and we verified that it works, we need to put it on Cloud
Storage so that ML Engine can access the code when running the training job. To do that we'll
place it in the Cloud Storage Bucket you created.
```
gsutil cp $PACKAGE_NAME gs://$BUCKET_ID
```

# Part 4: Submit your training job
Creates a config file for your job request. This tells ML Engine where to find the published
container.
```
cat > config.yaml <<EOF
# config.yaml
---
trainingInput:
  scaleTier: BASIC
  masterConfig:
    imageUri: $PUBLISHED_IMAGE_URI
EOF
```
Submit the training job to Cloud ML Engine using `gcloud`.

`module_name` and `package_uris` are expected by the published docker image to find your trainer
package and run your training job.

Note: You may need to install gcloud alpha to submit the training job.
```
gcloud components install alpha
```
```
gcloud alpha ml-engine jobs submit training $JOB_NAME \
  --region $REGION \
  --config=config.yaml \
  -- \
  --module_name=$TRAINER_MODULE \
  --package_uris=gs://$BUCKET_ID/$PACKAGE_NAME \
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
