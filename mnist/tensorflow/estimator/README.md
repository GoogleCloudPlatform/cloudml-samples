# Image classification for MNIST Dataset
_**Multi-class classification for Images**_

### Overview

This is an example of multi-class classification. This guide trains a neural
network model to classify images. This model was taken from the official
Tensorflow models Github [site](https://github.com/tensorflow/models/tree/master/official/mnist) and is adapted to run in CMLE engine.

### Tensorflow model in CMLE

This model builds a convolutional neural net to classify the MNIST dataset using the tf.data, tf.estimator.Estimator, and tf.Keras APIs.

## Dataset

We'll use the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/)
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. 
It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal 
efforts on preprocessing and formatting.

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
    ├── mnist.py
    ├── mnist_eager.py
    ├── dataset.py
    └── utils
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

### Adapting your Tensorflow code to run in CMLE

In this example we clone an existing model from TensorFlow models [repo](https://github.com/tensorflow/models)

Under the official folder you will find the MNIST [example](https://github.com/tensorflow/models/tree/master/official/mnist).
The models in this repository utilize as dependency the utils [subfolder](https://github.com/tensorflow/models/tree/master/official/utils)
hence is important to edit the references.

Instructions below:

Clone TensorFlow repository

```
git clone https://github.com/tensorflow/models.git
```

Create a new directory structure for CMLE:

```
mkdir -p mnist/trainer/utils
```

Copy utils folder: 

```
cd models/official/utils
cp -rv . ../../../mnist/trainer/utils/
```

Copy mnist.py, mnist_eager.py and dataset.py from MNIST [example](https://github.com/tensorflow/models/tree/master/official/mnist).
Note: The utils folder is not imported into this folder. You need to import it manually to your local folder.

```
cd ../mnist
cp -r dataset.py mnist.py mnist_eager.py ../../../mnist/trainer/
```


Edit references

You will notice files in folder contain references to the original Tensorflow github folder:
```
from official.utils.flags import _base
from official.utils.flags import _benchmark
from official.utils.flags import _conventions
from official.utils.flags import _device
from official.utils.flags import _misc
from official.utils.flags import _performance
```


Use your favorite text editor to replace ```from official.``` to ```from ```.
Then you need to convert to in all files. Example:
```
from utils.flags import _base
from utils.flags import _benchmark
from utils.flags import _conventions
from utils.flags import _device
from utils.flags import _misc
from utils.flags import _performance
```


To be able to import the utils folder, add the following lines to your entry point ```mnist.py``` or ```mnist_eager.py```

```
# Setup PYTHONPATH
import sys
import os
# Append current directory to the python path.
sys.path.append(os.path.dirname(__file__))
```


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

## Flags

Defined in flags.utils._base.py

| Parameter     |    Description        | Required  |
|:------------- |:-------------| -----:|
|data_dir|Create a flag for specifying the input data directory. | Yes |
|model_dir|Create a flag for specifying the model file directory. | Yes |
|train_epochs|Create a flag to specify the number of training epochs. | No |
|epochs_between_evals|Create a flag to specify the frequency of testing. | No |
|stop_threshold|Create a flag to specify a threshold accuracy or other eval metric which should trigger the end of training. | No |
|batch_size|Create a flag to specify the batch size. | No |
|num_gpu|Create a flag to specify the number of GPUs used. | No |
|hooks|Create a flag to specify hooks for logging. | No |
|export_dir|Create a flag to specify where a SavedModel should be exported. | No |

   
## Run the model locally

You can run the TensorFlow code locally to validate your project.

Define variables:

```
MNIST_DATA=data
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=mnist_$DATE
rm -rf $JOB_DIR
```

Run the model with python (local)

```
python -m trainer.mnist \
    --data_dir=$MNIST_DATA \
    --model_dir=$JOB_DIR \
    --export_dir=$JOB_DIR
```

*Note: To run mnist_eager.py just replace trainer.mnist with trainer.mnist_eager

## Training using gcloud local

You can run Keras training using gcloud locally.

Define variables*

```
export BUCKET_NAME=your-bucket-name
export JOB_NAME="mnist_keras_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export MNIST_DATA=gs://cloud-samples-data/ml-engine/mnist/
```

You can run Keras training using gcloud locally.

```
gcloud ml-engine local train --module-name=trainer.mnist --package-path=trainer -- \
    --data_dir=$MNIST_DATA \
    --model_dir=$JOB_DIR \
    --export_dir=$JOB_DIR \
    --train_epochs=1
```

*Note: To run mnist_eager.py just replace trainer.mnist with trainer.mnist_eager


## Training using Cloud ML Engine

You can train the model on Cloud ML Engine

Define variables

```
export BUCKET_NAME=your-bucket-name
export JOB_NAME="mnist_keras_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export MNIST_DATA=gs://cloud-samples-data/ml-engine/mnist/
```

You can train the model on Cloud ML Engine

```
gcloud ml-engine jobs submit training $JOB_NAME --stream-logs --runtime-version 1.10 \
    --staging-bucket=gs://$BUCKET_NAME \
    --package-path=trainer \
    --module-name trainer.mnist \
    --region $REGION -- \
    --data_dir=$MNIST_DATA \
    --model_dir=$JOB_DIR \
    --export_dir=$JOB_DIR
```
*Note: To run mnist_eager.py just replace trainer.mnist with trainer.mnist_eager

## Monitor training with TensorBoard

If Tensorboard appears blank, try refreshing after 10 minutes.

```
tensorboard --logdir=$JOB_DIR
```

## References

[Tensorflow tutorial](https://www.tensorflow.org/tutorials/keras/basic_classification).