# Predict House Prices

_**Logistic Regression**_

### Overview

This is an example of a regression problem. This guide trains a neural network
model to predict house prices based on different features.

### Keras model in CMLE

This sample runs both as a standalone Keras code and on Cloud ML Engine.

We aim to predict the output of a continuous value, like a price or a
probability.

## Dataset

We'll use the
[Boston Housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
This dataset contains information collected by the U.S Census Service concerning
housing in the area of Boston Mass. It was obtained from the StatLib archive
(http://lib.stat.cmu.edu/datasets/boston), and has been used extensively
throughout the literature to benchmark algorithms. The dataset
is small in size with only 506 cases.

The dataset contains 13 different features:

1.  Per capita crime rate.
2.  The proportion of residential land zoned for lots over 25,000 square feet.
3.  The proportion of non-retail business acres per town.
4.  Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
5.  Nitric oxides concentration (parts per 10 million).
6.  The average number of rooms per dwelling.
7.  The proportion of owner-occupied units built before 1940.
8.  Weighted distances to five Boston employment centers.
9.  Index of accessibility to radial highways.
10. Full-value property-tax rate per $10,000.
11. Pupil-teacher ratio by town.
12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
13. Percentage lower status of the population.

## How to satisfy Cloud ML Engine project structure requirements

The basic project structure will look something like this:

```shell
.
├── README.md
├── requirements.txt
├── setup.py
└── trainer
    ├── __init__.py
    ├── model.py
    └── task.py
```

### (Prerequisite) Set up and test your GCP environment

The best way to setup your GCP project is to use this section in this
[tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#set-up-your-gcp-project).

Once that the Cloud SDK is set up, you can check your Cloud ML Engine available
models by running: `gcloud ml-engine models list` You should see `Listed 0
items.` because we haven't created any ML Engine models yet.

## Virtual environment

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:

*   Install [Virtual](https://virtualenv.pypa.io/en/stable/) env
    *   Create virtual environment `virtualenv mnist`
    *   Activate env `source mnist/bin/activate`
*   Install [Miniconda](https://conda.io/miniconda.html)
    *   Create conda environment `conda create --name mnist python=2.7`
    *   Activate env `source activate mnist`

## Install dependencies

*   Install the python dependencies. `pip install --upgrade -r requirements.txt`

### Get your training data

The code from the Tensorflow website
[Predict house prices: regression](https://www.tensorflow.org/tutorials/keras/basic_regression)
downloads the Boston data *every time* it is run. The Boston dataset comes
packaged with TensorFlow. Downloading the data is impractical/expensive for
large datasets, so we will get the original files to illustrate a [more general
data preparation process] you might follow in your own projects.

If you want to use local files directly, you can use the following commands:

```shell
mkdir data && cd data
curl -O https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz
cd ..
```

### Upload the data to a Google Cloud Storage bucket

Cloud ML Engine works by using resources available in the cloud, so the training
data needs to be placed in such a resource. For this example, we'll use [Google
Cloud Storage], but it's possible to use other resources like [BigQuery]. Make a
bucket (names must be globally unique) and place the data in there:

```shell
gsutil mb gs://your-bucket-name
gsutil cp -r data/boston_housing.npz gs://your-bucket-name/boston_housing.npz
```

### Project configuration file: `setup.py`

The `setup.py` file is run on the Cloud ML Engine server to install
packages/dependencies and set a few options.

```python
from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
  requirements = [l.strip('\n') for l in f if
                  l.strip('\n') and not l.startswith('#')]
                  
setup(
  name='boston',
  version='0.1',
  install_requires=requirements,
  packages=find_packages(),
  include_package_data=True,
  description='CMLE samples'
)
```

Technically, Cloud ML Engine [requires a TensorFlow application to be
pre-packaged] so that it can install it on the servers it spins up. However, if
you supply a `setup.py` in the project root directory, then Cloud ML Engine will
actually create the package for you.

### Create the `__init__.py` file

For the Cloud ML Engine to create a package for your module, it's absolutely for
your project to contain `trainer/__init__.py`, but it can be empty.

```shell
mkdir trainer
touch trainer/__init__.py`
```

Without `__init__.py` local training will work, but when you try to submit a job
to Cloud ML Engine, you will get the cryptic error message:

```shell
ERROR: (gcloud.ml-engine.jobs.submit.training)
[../trainer] is not a valid Python package because it does not contain an
`__init__.py` file. Please create one and try again.
```

## Run the model locally

You can run the Keras code locally to validate your project.

Use local training file.

```
export DATA_FOLDER=data
mkdir $DATA_FOLDER
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME="boston_keras_$DATE"
export JOB_DIR=boston_$DATE
export TRAIN_FILE=$DATA_FOLDER/boston_housing.npz
export JOB_DIR=/tmp/$JOB_NAME
export REGION=us-central1
rm -rf $JOB_DIR
```

Use remote file located in GCS.

```
export BUCKET_NAME=your-bucket-name
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME="boston_keras_$DATE"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/boston/boston_housing.npz
export REGION=us-central1
rm -rf $JOB_DIR
```

Run the model with python (local)

```
python -m trainer.task --train_file=$TRAIN_FILE --job_dir=$JOB_DIR
```

## Training using gcloud local

You can run Keras training using gcloud locally.

Define variables*

```
export BUCKET_NAME=your-bucket-name
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME="boston_keras_$DATE"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/boston/boston_housing.npz
```

You can run Keras training using gcloud locally.

```
gcloud ml-engine local train --module-name=trainer.task --package-path=trainer -- --train_file=$DATASET_FILE --job_dir=$JOB_DIR
```

*Feel free to modify the destination file for in utils.py

## Training using Cloud ML Engine

You can train the model on Cloud ML Engine

Define variables

```
export BUCKET_NAME=your-bucket-name
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME="boston_keras_$DATE"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/boston/boston_housing.npz
```

You can train the model on Cloud ML Engine

```
gcloud ml-engine jobs submit training $JOB_NAME --stream-logs --runtime-version 1.10 --job_dir=$JOB_DIR --package-path=trainer --module-name trainer.task --region $REGION -- --train_file=$DATASET_FILE
```

## Monitor training with TensorBoard

If Tensorboard appears blank, try refreshing after 10 minutes.

```
tensorboard --logdir=$JOB_DIR
```

## References

[Tensorflow tutorial](https://www.tensorflow.org/tutorials/keras/basic_regression).
