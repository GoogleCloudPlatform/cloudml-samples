# IMDB Text Classification

Binary classification

- - -

This is an example of binary or two-class classification, an important and
widely applicable kind of machine learning problem.
This example classifies movie reviews as positive or negative using the text of
the review.
This sample code is based on Tensorflow
[tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification).

### Examples

* The sample provided in [Keras](./tensorflow/keras) uses the Keras library inside Tensorflow.
  This API is great for fast iteration, and quickly adapting models to your own datasets 
  without major code overhauls.
 
All the models provided in this directory can be run on the Cloud Machine Learning Engine. To follow along, check out the setup instructions [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

## Dataset
We'll use the
[IMDB dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)
that contains the text of 50,000 movie reviews from the
[Internet Movie Database](https://www.imdb.com/). These are split into 25,000
reviews for training and 25,000 reviews for testing. The training and testing
sets are balanced, meaning they contain an equal number of positive and negative
reviews.

### Disclaimer
This dataset is provided by a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.

## Setup Instructions

In order to satisfy Cloud ML Engine project structure requirements. The basic project structure will look something like this:

```shell
.
├── README.md
├── setup.py
└── trainer
    ├── __init__.py
    ├── model.py
    └── task.py
```
All the models provided in this directory can be run on the Cloud Machine Learning Engine. To follow along, check out the setup instructions [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).


## Virtual environment

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:

*   Install [Virtual](https://virtualenv.pypa.io/en/stable/) env
    *   Create virtual environment `virtualenv imdb_keras`
    *   Activate env `source imdb_keras/bin/activate`
*   Install [Miniconda](https://conda.io/miniconda.html)
    *   Create conda environment `conda create --name imdb_keras python=2.7`
    *   Activate env `source activate imdb_keras`

## Install dependencies

*   Install the python dependencies. `pip install --upgrade -r requirements.txt`

### Get your training data

The code from the Keras github
[IMDB example](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
downloads the IMDB data *every time* it is run. The IMDB dataset comes packaged
with TensorFlow. Downloading the data is impractical/expensive for large
datasets, so we will get a pre-processed version of the IMDB data to illustrate
a [more general data preparation process] you might follow in your own projects.

The file `imdb.npz` has already been preprocessed using Numpy such that the
reviews (sequences of words) have been converted to sequences of integers, where
each integer represents a specific word in a dictionary. This dictionary object
contains the integer to string mapping `imdb_word_index.json`.

```shell
mkdir data
curl -O https://s3.amazonaws.com/text-datasets/imdb.npz
curl -O https://s3.amazonaws.com/text-datasets/imdb_word_index.json
```

### Upload the data to a Google Cloud Storage bucket

Cloud ML Engine works by using resources available in the cloud, so the training
data needs to be placed in such a resource. For this example, we'll use [Google
Cloud Storage], but it's possible to use other resources like [BigQuery]. Make a
bucket (names must be globally unique) and place the data in there:

```shell
gsutil mb gs://your-bucket-name
gsutil cp -r data/imdb.npz gs://your-bucket-name/imdb.npz
gsutil cp -r data/imdb_word_index.json gs://your-bucket-name/imdb_word_index.json
```

### Project configuration file: `setup.py`

The `setup.py` file is run on the Cloud ML Engine server to install
packages/dependencies and set a few options.

```python
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['requests==2.19.1']

setup(name='imdb',
      version='1.0',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=find_packages(),
      description='IMDB Keras model on Cloud ML Engine'
)
```

Technically, Cloud ML Engine [requires a TensorFlow application to be
pre-packaged] so that it can install it on the servers it spins up. However, if
you supply a `setup.py` in the project root directory, then Cloud ML Engine will
actually create the package for you.

## Run the model locally

Run the code on your local machine:

Define variables

```
IMDB_DATA=data
mkdir $IMDB_DATA
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=imdb_$DATE
rm -rf $JOB_DIR
```

Run the model with python (local)
                       
```
python -m trainer.task --train-file=$IMDB_DATA/imdb.npz \
                       --word-index-file=$IMDB_DATA/imdb_word_index.json \
                       --job-dir=$JOB_DIR
```

## Training using gcloud local

Run the code on your local machine using `gcloud`. This allows you to "mock"
running it on the Google Cloud:

Define variables*

```
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=imdb_$DATE
rm -rf $JOB_DIR
export REGION=us-central1
export GCS_TRAIN_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb.npz
export GCS_WORD_INDEX_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb_word_index.json
```

You can run Keras training using gcloud locally.

```
gcloud ml-engine local train --module-name=trainer.task \
                --package-path=trainer/ \
                -- \
                --train-file=$GCS_TRAIN_FILE \
                --word-index-file=$GCS_WORD_INDEX_FILE \
                --job-dir=$JOB_DIR
```

*Feel free to modify the destination file for in utils.py

## Training using Cloud ML Engine

Run the code on Cloud ML Engine using `gcloud`. Note how `--job-dir` comes
before `--` while training on the cloud and this is so that we can have
different trial runs during Hyperparameter tuning.

```
export BUCKET_NAME=your-bucket-name-without-gcs-prefix
export JOB_NAME="imbd_keras_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export GCS_TRAIN_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb.npz
export GCS_WORD_INDEX_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb_word_index.json
```

You can train the model on Cloud ML Engine

```
gcloud ml-engine jobs submit training $JOB_NAME \
                --stream-logs \
                --runtime-version 1.10 \
                --job-dir=$JOB_DIR \
                --package-path=trainer \
                --module-name trainer.task \
                --region $REGION \
                -- \
                --train-file $GCS_TRAIN_FILE \
                --word-index-file $GCS_WORD_INDEX_FILE             
```

## Monitor training with TensorBoard

If Tensorboard appears blank, try refreshing after 10 minutes.

```
tensorboard --logdir=$JOB_DIR
```
