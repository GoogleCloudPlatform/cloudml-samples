<h1>Overview</h1>
This code implements a Binary or two-classification model using the Google Cloud Platform. It includes code to process data, 
train a TensorFlow model and assess model performance.
This example classifies movie reviews as positive or negative using the text of the review.

#
* **Data description**

We'll use the
[IMDB dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)
that contains the text of 50,000 movie reviews from the
[Internet Movie Database](https://www.imdb.com/). These are split into 25,000
reviews for training and 25,000 reviews for testing. The training and testing
sets are balanced, meaning they contain an equal number of positive and negative
reviews.

* **Disclaimer**

This dataset is provided by a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.

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

* **Upload the data to a Google Cloud Storage bucket**

Cloud ML Engine works by using resources available in the cloud, so the training
data needs to be placed in such a resource. For this example, we'll use [Google
Cloud Storage], but it's possible to use other resources like [BigQuery]. Make a
bucket (names must be globally unique) and place the data in there:

```shell
gsutil mb gs://your-bucket-name
gsutil cp -r data/imdb.npz gs://your-bucket-name/imdb.npz
gsutil cp -r data/imdb_word_index.json gs://your-bucket-name/imdb_word_index.json
```

<h1>Training</h1>

* **GCloud configuration:**

```
IMDB_DATA=data
mkdir $IMDB_DATA
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=imdb_$DATE
export TRAIN_FILE=$IMDB_DATA/imdb.npz
export WORD_INDEX_FILE=$IMDB_DATA/imdb_word_index.json
rm -rf $JOB_DIR
```

* **Test locally:**

```
python -m trainer.task --train-file=$TRAIN_FILE \
    --word-index-file=$WORD_INDEX_FILE \
    --job-dir=$JOB_DIR
```

* **Google Cloud ML Engine:**

* **GCloud configuration:**

```
export BUCKET_NAME=your-bucket-name
export JOB_NAME="imbd_keras_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb.npz
export WORD_INDEX_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb_word_index.json
```

* **Run locally:**

```
gcloud ml-engine local train --module-name=trainer.task \
    --package-path=trainer/ \
    -- \
    --train-file=$TRAIN_FILE \
    --word-index-file=$WORD_INDEX_FILE \
    --job-dir=$JOB_DIR
```

*Feel free to modify the destination file for in utils.py

* **Run in Google Cloud ML Engine:**

You can train the model on Cloud ML Engine:


```
gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --runtime-version 1.10 \
    --job-dir=$JOB_DIR \
    --package-path=trainer \
    --module-name trainer.task \
    --region $REGION \
    -- \
    --train-file $TRAIN_FILE \
    --word-index-file $WORD_INDEX_FILE             
```

* **Monitor with TensorBoard:**

```
tensorboard --logdir=$JOB_DIR
```

## References

[Tensorflow tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
