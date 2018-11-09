# Image classification for MNIST Fashion Dataset.
_**Multi-class classification for Images**_

### Overview

This is an example of multi-class classification. This guide trains a neural
network model to classify images of clothing, like sneakers and shirts.

### Keras model in CMLE

This sample runs both as a standalone Keras code and on Cloud ML Engine.

Fashion MNIST is intended as a drop-in replacement for the classic MNIST
dataset—often used as the "Hello, World" of machine learning programs for
computer vision. The MNIST dataset contains images of handwritten digits (0, 1,
2, etc) in an identical format to the articles of clothing we'll use here.

This guide uses Fashion MNIST for variety, and because it's a slightly more
challenging problem than regular MNIST. Both datasets are relatively small and
are used to verify that an algorithm works as expected. They're good starting
points to test and debug code.

## Dataset

We'll use the
[Fashion dataset](https://github.com/zalandoresearch/fashion-mnist)
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training
set of 60,000 examples and a test set of 10,000 examples. Each example is a
28x28 grayscale image, associated with a label from 10 classes.

We will use 60,000 images to train the network and 10,000 images to evaluate how
accurately the network learned to classify images

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
    ├── task.py
    └── utils.py
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

The code from the Keras github
[MNIST Fashion example](https://www.tensorflow.org/tutorials/keras/basic_classification)
downloads the MNIST data *every time* it is run. The MNIST dataset comes
packaged with TensorFlow. Downloading the data is impractical/expensive for
large datasets, so we will get the original files to illustrate a [more general
data preparation process] you might follow in your own projects.

If you want to download the files directly, you can use the following commands:

```shell
mkdir data && cd data
curl -O https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
```

### Upload the data to a Google Cloud Storage bucket

Cloud ML Engine works by using resources available in the cloud, so the training
data needs to be placed in such a resource. For this example, we'll use [Google
Cloud Storage], but it's possible to use other resources like [BigQuery]. Make a
bucket (names must be globally unique) and place the data in there:

```shell
gsutil mb gs://your-bucket-name
gsutil cp -r data/train-labels-idx1-ubyte.gz gs://your-bucket-name/train-labels-idx1-ubyte.gz
gsutil cp -r data/train-images-idx3-ubyte.gz gs://your-bucket-name/train-images-idx3-ubyte.gz
gsutil cp -r data/train-labels-idx1-ubyte.gz gs://your-bucket-name/t10k-images-idx3-ubyte.gz
gsutil cp -r data/train-images-idx3-ubyte.gz gs://your-bucket-name/t10k-labels-idx1-ubyte.gz
```

### Project configuration file: `setup.py`

The `setup.py` file is run on the Cloud ML Engine server to install
packages/dependencies and set a few options.


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

Define variables

```
MNIST_DATA=data
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=mnist_$DATE
export TRAIN_FILE=$MNIST_DATA/train-images-idx3-ubyte.gz
export TRAIN_LABELS_FILE=$MNIST_DATA/train-labels-idx1-ubyte.gz
export TEST_FILE=$MNIST_DATA/train-images-idx3-ubyte.gz
export TEST_LABELS_FILE=$MNIST_DATA/train-labels-idx1-ubyte.gz
rm -rf $JOB_DIR
```

Run the model with python (local)

```
python -m trainer.task \
    --train-file=$TRAIN_FILE \
    --train-labels-file=$TRAIN_LABELS_FILE \
    --test-file=$TEST_FILE \
    --test-labels-file=$TEST_LABELS_FILE \
    --job-dir=$JOB_DIR
```

## Training using gcloud local

You can run Keras training using gcloud locally.

Define variables*

```
export BUCKET_NAME=your-bucket-name
export JOB_NAME="mnist_keras_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/mnist/train-images-idx3-ubyte.gz
export TRAIN_LABELS_FILE=gs://cloud-samples-data/ml-engine/mnist/train-labels-idx1-ubyte.gz
export TEST_FILE=gs://cloud-samples-data/ml-engine/mnist/t10k-images-idx3-ubyte.gz
export TEST_LABELS_FILE=gs://cloud-samples-data/ml-engine/mnist/t10k-labels-idx1-ubyte.gz
```

You can run Keras training using gcloud locally.

```
gcloud ml-engine local train --module-name=trainer.task --package-path=trainer -- \
    --train-file=$TRAIN_FILE \
    --train-labels=$TRAIN_LABELS_FILE \
    --test-file=$TEST_FILE \
    --test-labels_file=$TEST_LABELS_FILE \
    --job-dir=$JOB_DIR
```

*Feel free to modify the destination file for in utils.py

## Training using Cloud ML Engine

You can train the model on Cloud ML Engine

Define variables

```
export BUCKET_NAME=your-bucket-name
export JOB_NAME="mnist_keras_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/mnist/train-images-idx3-ubyte.gz
export TRAIN_LABELS_FILE=gs://cloud-samples-data/ml-engine/mnist/train-labels-idx1-ubyte.gz
export TEST_FILE=gs://cloud-samples-data/ml-engine/mnist/t10k-images-idx3-ubyte.gz
export TEST_LABELS_FILE=gs://cloud-samples-data/ml-engine/mnist/t10k-labels-idx1-ubyte.gz
```

You can train the model on Cloud ML Engine

```
gcloud ml-engine jobs submit training $JOB_NAME --stream-logs --runtime-version 1.10 \
    --job-dir=$JOB_DIR \
    --package-path=trainer \
    --module-name trainer.task \
    --region $REGION -- \
    --train-file=$TRAIN_FILE \
    --train-labels=$TRAIN_LABELS_FILE \
    --test-file=$TEST_FILE \
    --test-labels-file=$TEST_LABELS_FILE
```

## Monitor training with TensorBoard

If Tensorboard appears blank, try refreshing after 10 minutes.

```
tensorboard --logdir=$JOB_DIR
```

## References

[Tensorflow tutorial](https://www.tensorflow.org/tutorials/keras/basic_classification).
