<h1>Overview</h1>
This code implements a Multi-class classification model using the Google Cloud Platform. It includes code to process data, train a TensorFlow model and assess model performance.
This guide trains a neural network model to classify images of clothing, like sneakers and shirts.

#
* **Data description**

We'll use the
[Fashion dataset](https://github.com/zalandoresearch/fashion-mnist)
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training
set of 60,000 examples and a test set of 10,000 examples. Each example is a
28x28 grayscale image, associated with a label from 10 classes.

We will use 60,000 images to train the network and 10,000 images to evaluate how
accurately the network learned to classify images.

* **Set up and test your GCP environment**

The best way to setup your GCP project is to use this section in this
[tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#set-up-your-gcp-project).

* **Environment set-up:**

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:

*   Install [Virtualenv](https://virtualenv.pypa.io/en/stable/) 
    *   Create virtual environment `virtualenv myvirtualenv`
    *   Activate env `source myvirtualenv/bin/activate`
*   Install [Miniconda](https://conda.io/miniconda.html)
    *   Create conda environment `conda create --name myvirtualenv python=2.7`
    *   Activate env `source activate myvirtualenv`

* **Install dependencies**

Install the python dependencies. `pip install --upgrade -r requirements.txt`

# 
* **How to satisfy Cloud ML Engine project structure requirements**

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

<h1>Data processing</h1>

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

* **Upload the data to a Google Cloud Storage bucket**

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

<h1>Training</h1>

* **GCloud configuration:**

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

* **Run locally:**

```
python -m trainer.task \
    --train-file=$TRAIN_FILE \
    --train-labels-file=$TRAIN_LABELS_FILE \
    --test-file=$TEST_FILE \
    --test-labels-file=$TEST_LABELS_FILE \
    --job-dir=$JOB_DIR
```

* **Run locally in Google Cloud ML Engine:**

* **GCloud configuration:**

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

```
gcloud ml-engine local train --module-name=trainer.task --package-path=trainer -- \
    --train-file=$TRAIN_FILE \
    --train-labels=$TRAIN_LABELS_FILE \
    --test-file=$TEST_FILE \
    --test-labels_file=$TEST_LABELS_FILE \
    --job-dir=$JOB_DIR
```

* **Run in Google Cloud ML Engine:**

You can train the model on Cloud ML Engine:

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

* **Monitor with TensorBoard:**

```
tensorboard --logdir=$JOB_DIR
```

## References

[Tensorflow tutorial](https://www.tensorflow.org/tutorials/keras/basic_classification).
