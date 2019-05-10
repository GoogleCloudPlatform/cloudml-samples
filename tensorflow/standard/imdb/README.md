# Overview
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

* **Setup and test your GCP environment**

The best way to setup your GCP project is to use this section in this
[tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#set-up-your-gcp-project).

* **Environment setup:**

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
locally without changing global python packages on your system.

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
* **How to satisfy AI Platform project structure requirements**

Follow [this](https://cloud.google.com/ml-engine/docs/tensorflow/packaging-trainer#project-structure) guide to structure your training application.


# Data processing

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
gsutil cp gs://cloud-samples-data/ml-engine/imdb/imdb.npz data
gsutil cp gs://cloud-samples-data/ml-engine/imdb/imdb_word_index.json data
```

* **Upload the data to a Google Cloud Storage bucket**

AI Platform works by using resources available in the cloud, so the training
data needs to be placed in such a resource. For this example, we'll use [Google
Cloud Storage], but it's possible to use other resources like [BigQuery]. Make a
bucket (names must be globally unique) and place the data in there:

```shell
gsutil mb gs://your-bucket-name
gsutil cp -r data/imdb.npz gs://your-bucket-name/imdb.npz
gsutil cp -r data/imdb_word_index.json gs://your-bucket-name/imdb_word_index.json
```

# Training

* **GCloud configuration:**

```
IMDB_DATA=data
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

* **AI Platform:**

* **GCloud configuration:**

```
export BUCKET_NAME=your-bucket-name
export JOB_NAME="imbd_keras_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb.npz
export WORD_INDEX_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb_word_index.json
```

* **Run locally via the gcloud command for AI Platform:**

```
gcloud ml-engine local train --module-name=trainer.task \
    --package-path=trainer/ \
    -- \
    --train-file=$TRAIN_FILE \
    --word-index-file=$WORD_INDEX_FILE \
    --job-dir=$JOB_DIR
```

* **Run in AI Platform**

You can train the model on AI Platform:

*NOTE:* If you downloaded the training files to your local filesystem, be sure
to reset the `TRAIN_FILE` and `WORD_INDEX_FILE` environment variables to refer to a GCS location.
Data must be in GCS for cloud-based training.

Run the code on AI Platform using `gcloud`. Note how `--job-dir` comes
before `--` while training on the cloud and this is so that we can have
different trial runs during Hyperparameter tuning.

* **GCloud configuration:**

```
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=imdb_$DATE
export GCS_JOB_DIR=gs://your-bucket-name/path/to/my/jobs/$JOB_NAME
echo $GCS_JOB_DIR
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb.npz
export WORD_INDEX_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb_word_index.json
export REGION=us-central1
```

* **Run in AI Platform:**

```
gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --runtime-version 1.10 \
    --job-dir=$GCS_JOB_DIR \
    --package-path=trainer \
    --module-name trainer.task \
    --region $REGION \
    -- \
    --train-file $TRAIN_FILE \
    --word-index-file $WORD_INDEX_FILE             
```

* **Monitor with TensorBoard:**

```
tensorboard --logdir=$GCS_JOB_DIR
```

## References

[Tensorflow tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
