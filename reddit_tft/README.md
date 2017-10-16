## Reddit Sample

The Reddit sample demonstrates the capability of both linear and deep models on
a [Reddit Dataset](https://www.reddit.com/r/bigquery/wiki/datasets).

## Prerequisites

*   Make sure you follow the Google Cloud ML setup
    [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up)
    before trying the sample. More documentation about Cloud ML is available
    [here](https://cloud.google.com/ml/docs/).
*   Make sure your Google Cloud project has sufficient quota.

## Install Dependencies

Install dependencies by running `pip install -r requirements.txt`

## Sample Overview

This sample consists of two parts:

### Data Pre-Processing

Data pre-processing step involves reading data from Google Cloud BigQuery
and converting it to
[TFRecords](https://www.tensorflow.org/api_guides/python/python_io)
format.

### Model Training

Model training step involves taking the pre-processed TFRecords data and
training a linear classifier using Stochastic Dual Coordinate Ascent (SDCA)
optimizer, or a deep neural network classifier.

## Reddit Comments Dataset

[Multiple years'](https://bigquery.cloud.google.com/dataset/fh-bigquery:reddit_comments)
worth of Reddit Comments are publicly available in Google Cloud BigQuery. We
will use a subset of the data and some SQL manipulation to create training data
for predicting the score of a Reddit thread.

## Data Format

Above dataset is available in BigQuery and need to be transformed to
TFRecords format for the sample code to work. Make sure to run the data through
the pre-processing step before you proceed to training.

## Pre-Processing Step

The pre-processing step can be performed either locally or on cloud depending
upon the size of input data.

First you need to separate your input into training and evaluation sets. We can
use one month's worth of data
([December 2015](https://bigquery.cloud.google.com/table/fh-bigquery:reddit_comments.2015_12))
which amounts to approximately 20GB for training and we can then evaluate
ourselves on a month's worth of "future" data
([January 2016](https://bigquery.cloud.google.com/table/fh-bigquery:reddit_comments.2016_01)).
Finally we can issue predictions for data even "further in the future"
([February 2016](https://bigquery.cloud.google.com/table/fh-bigquery:reddit_comments.2016_02)).

We use the appropriate table names as the input flags `--training_data`,
`--eval_data` and `--predict_data` respectively.

### Cloud Run

In order to run pre-processing on the Cloud run the commands below.

```
PROJECT=$(gcloud config list project --format "value(core.project)")
BUCKET="gs://${PROJECT}-ml"

GCS_PATH="${BUCKET}/${USER}/reddit_comments"
```

```
PREPROCESS_OUTPUT="${GCS_PATH}/reddit_$(date +%Y%m%d_%H%M%S)"
python preprocess.py --training_data fh-bigquery.reddit_comments.2015_12 \
                     --eval_data fh-bigquery.reddit_comments.2016_01 \
                     --predict_data fh-bigquery.reddit_comments.2016_02 \
                     --output_dir "${PREPROCESS_OUTPUT}" \
                     --project_id "${PROJECT}" \
                     --cloud
```

## Models

The sample implements a linear model trained with SDCA, as well a deep neural
network model. The code can be run either locally or on cloud.

### Cloud Run

#### Help options

```
  python -m trainer.task -h
```

#### Train

To train the linear model (with crosses):

```
JOB_ID="reddit_comments_linear_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --stream-logs \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config-small.yaml \
  -- \
  --model_type linear \
  --l2_regularization 3000 \
  --eval_steps 1000 \
  --output_path "${GCS_PATH}/model/${JOB_ID}" \
  --raw_metadata_path "${PREPROCESS_OUTPUT}/raw_metadata" \
  --transformed_metadata_path "${PREPROCESS_OUTPUT}/transformed_metadata" \
  --transform_savedmodel "${PREPROCESS_OUTPUT}/transform_fn" \
  --eval_data_paths "${PREPROCESS_OUTPUT}/features_eval*" \
  --train_data_paths "${PREPROCESS_OUTPUT}/features_train*"
```

To train the deep model:

```
JOB_ID="reddit_comments_deep_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --stream-logs \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config-small.yaml \
  -- \
  --model_type deep \
  --hidden_units 1062 1062 1062 1062 1062 1062 1062 1062 1062 1062 1062 \
  --batch_size 512 \
  --eval_steps 250 \
  --output_path "${GCS_PATH}/model/${JOB_ID}" \
  --raw_metadata_path "${PREPROCESS_OUTPUT}/raw_metadata" \
  --transformed_metadata_path "${PREPROCESS_OUTPUT}/transformed_metadata" \
  --transform_savedmodel "${PREPROCESS_OUTPUT}/transform_fn" \
  --eval_data_paths "${PREPROCESS_OUTPUT}/features_eval*" \
  --train_data_paths "${PREPROCESS_OUTPUT}/features_train*"
```
