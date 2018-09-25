# Text classification with Movie Reviews

This is an example of binary or two-class classification, an important and
widely applicable kind of machine learning problem. This sample code is
originally taken from Tensorflow
[tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification).

### Keras model in CMLE

This sample runs both as a standalone Keras code and on Cloud ML Engine.

This example classifies movie reviews as positive or negative using the text of
the review.

We'll use the
[IMDB dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)
that contains the text of 50,000 movie reviews from the
[Internet Movie Database](https://www.imdb.com/). These are split into 25,000
reviews for training and 25,000 reviews for testing. The training and testing
sets are balanced, meaning they contain an equal number of positive and negative
reviews.

## How to satisfy Cloud ML Engine project structure requirements

The basic project structure will look something like this:

```shell
.
├── README.md
├── data
│   ├── imdb.npz
│   └── imdb_word_index.json
├── setup.py
└── trainer
    ├── __init__.py
    ├── model.py
    ├── task.py
    └── utils.py
```

### (Prerequisite) Set up and test your GCP environment

The best way to setup your GCP project is to use this section in this
[tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#set-up-your-gcp-project).

Now that the Cloud SDK is set up, you can check your Cloud ML Engine available
models: `shell gcloud ml-engine models list` You should see `Listed 0 items.`
because we haven't created any ML Engine models yet.

If you have installed `gcloud` previously, update `gcloud`:

```shell
gcloud components update
```

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

### Create the `__init__.py` file

For the Cloud ML Engine to create a package for your module, it's absolutely for
your project to contain `trainer/__init__.py`, but it can be empty. `shell mkdir
trainer touch trainer/__init__.py` Without `__init__.py` local training will
work, but when you try to submit a job to Cloud ML Engine, you will get the
cryptic error message: ``shell ERROR: (gcloud.ml-engine.jobs.submit.training)
[../trainer] is not a valid Python package because it does not contain an
`__init__.py` file. Please create one and try again.``

## Run the model locally

You can run the Keras code locally.

Define variables

```
export BUCKET_NAME=your-bucket-name
export JOB_NAME="imbd_keras_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
```

Run the model with python (local)

```
python -m trainer.task --train-file=data/imdb.npz \
                       --word-index-file=data/imdb_word_index.json \
                       --output_dir=/tmp/
```

## Training using gcloud local

You can run Keras training using gcloud locally.

Define variables*

```
export BUCKET_NAME=your-bucket-name
export JOB_NAME="imbd_keras_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export GCS_TRAIN_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb.npz
export GCS_WORD_INDEX_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb_word_index.json
```

You can run Keras training using gcloud locally.

```
gcloud ml-engine local train
                --module-name=trainer.task
                --package-path=trainer/
                --
                --train-file=$GCS_TRAIN_FILE
                --word-index-file=$GCS_WORD_INDEX_FILE
                --output_dir=$OUTPUT_DIR
```

*Feel free to modify the destination file for in utils.py

## Training using Cloud ML Engine

You can train the model on Cloud ML Engine

Define variables

```
export BUCKET_NAME=your-bucket-name
export JOB_NAME="imbd_keras_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export GCS_TRAIN_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb.npz
export GCS_WORD_INDEX_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb_word_index.json
```

You can train the model on Cloud ML Engine

```
gcloud ml-engine jobs submit training $JOB_NAME
                --stream-logs
                --runtime-version 1.10
                --job-dir=$OUTPUT_DIR
                --package-path=trainer
                --module-name trainer.task
                --region $REGION
                --
                --train-file $GCS_TRAIN_FILE
                --word-index-file $GCS_WORD_INDEX_FILE
                --output_dir=$OUTPUT_DIR
```

## Monitor training with TensorBoard

If Tensorboard appears blank, try refreshing after 10 minutes.

```
tensorboard --logdir=$OUTPUT_DIR
```
