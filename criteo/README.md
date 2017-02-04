## Criteo Sample

The Criteo sample demonstrates the capability of both linear and deep models on
the [criteo dataset](https://www.kaggle.com/c/criteo-display-ad-challenge).

## Prerequisites

*   Make sure you follow the cloud-ml setup
    [here](https://cloud.google.com/ml/docs/) before trying the
    sample.
*   Make sure you setup your environment
    [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).
*   Make sure you have installed
    [Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html).
*   Make sure you have installed
    [virtualenv](https://virtualenv.pypa.io/en/stable/installation/).
*   Make sure your Google Cloud project has sufficient quota.

## Sample Overview

This sample consists of two parts:

### Data Pre-Processing

Data pre-processing step involves taking the TSV data as input and converting it
to
[TFRecords](https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details)
format.

### Model Training

Model training step involves taking the pre-processed TFRecords data and
training a linear classifier using Stochastic Dual Coordinate Ascent (SDCA)
optimizer, or a deep neural network classifier.

## Criteo Dataset

The criteo dataset is available in two different sizes:

### Kaggle Challenge Dataset

Kaggle challenge dataset can be downloaded
[here](http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/).
We recommend working with the Kaggle dataset if you are trying this sample for
the first time. This dataset is about 10 GB and contains around 45 million
examples.

### Terabyte Click Logs

Terabyte click logs can be downloaded
[here](http://labs.criteo.com/downloads/download-terabyte-click-logs/). You
should use this dataset once you have experimented with the kaggle dataset. This
dataset is about 1 TB and contains 4 billion examples. Due to the sheer size of
this data, training with this dataset takes more time and is more expensive.

## Data Format

Above datasets are available in TSV format and need to be transformed to
TFRecords format for the sample code to work. Make sure to run the data through
the pre-processing step before you proceed to training.

## Pre-Processing Step

The pre-processing step can be performed either locally or on cloud depending
upon the size of input data.

First you need to split the tsv files into training and evaluation sets. Then
use those paths as the input flags `--training_data` and `--eval_data`
respectively.

### Local Run

Run the code as below:

```
python preprocess.py --training_data $LOCAL_TRAINING_INPUT \
                     --eval_data $LOCAL_EVAL_INPUT \
                     --output_dir $LOCAL_OUTPUT_DIR \
```

### Cloud Run

In order to run pre-processing on the Cloud run the commands below.

```
PROJECT=$(gcloud config list project --format "value(core.project)")
BUCKET="gs://${PROJECT}-ml"

# Small dataset
GCS_PATH="${BUCKET}/${USER}/smallclicks"
python preprocess.py --training_data $GCS_PATH/train.txt \
                     --eval_data $GCS_PATH/train.txt \
                     --output_dir $GCS_PATH/preproc \
                     --project_id $PROJECT \
                     --cloud

# Large dataset
GCS_PATH="${BUCKET}/${USER}/largeclicks"
python preprocess.py --training_data $GCS_PATH/day_* \
                     --eval_data $GCS_PATH/eval_day_* \
                     --output_dir $GCS_PATH/preproc \
                     --project_id $PROJECT \
                     --frequency_threshold 1000
                     --cloud
```

## Models

The sample implements a linear model trained with SDCA, as well a deep neural
network model. The code can be run either locally or on cloud.

### Local Run For the Small Dataset

Run the code as below:

#### Help options

```
  python -m trainer.task -h
```

#### How to run code

To train the linear model:

```
python -m trainer.task \
          --dataset kaggle \
          --l2_regularization 60 \
          --train_data_paths $LOCAL_OUTPUT_DIR/features_train* \
          --eval_data_paths $LOCAL_OUTPUT_DIR/features_eval* \
          --metadata_path $LOCAL_OUTPUT_DIR/metadata.json \
          --output_path $TRAINING_OUTPUT_PATH
```

To train the deep model:

```
python -m trainer.task \
          --dataset kaggle \
          --model_type deep \
          --hidden_units 600 600 600 600 \
          --batch_size 512 \
          --train_data_paths $LOCAL_OUTPUT_DIR/features_train* \
          --eval_data_paths $LOCAL_OUTPUT_DIR/features_eval* \
          --metadata_path $LOCAL_OUTPUT_DIR/metadata.json \
          --output_path $TRAINING_OUTPUT_PATH
```

### Cloud Run for the Small Dataset

You can train using either a single worker (config-single.yaml), or using
multiple workers and parameter servers (config-small.yaml).

To train the linear model:

```
JOB_ID="smallclicks_linear_${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud beta ml jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config-small.yaml \
  --async \
  -- \
  --dataset kaggle \
  --model_type linear \
  --l2_regularization 60 \
  --output_path "${GCS_PATH}/output/${JOB_ID}" \
  --metadata_path "${GCS_PATH}/preproc/metadata.json" \
  --eval_data_paths "${GCS_PATH}/preproc/features_eval*" \
  --train_data_paths "${GCS_PATH}/preproc/features_train*"
```

To train the deep model:

```
JOB_ID="smallclicks_deep_${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud beta ml jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config-small.yaml \
  --async \
  -- \
  --dataset kaggle \
  --model_type deep \
  --hidden_units 600 600 600 600 \
  --batch_size 512 \
  --output_path "${GCS_PATH}/output/${JOB_ID}" \
  --metadata_path "${GCS_PATH}/preproc/metadata.json" \
  --eval_data_paths "${GCS_PATH}/preproc/features_eval*" \
  --train_data_paths "${GCS_PATH}/preproc/features_train*"
```

### Cloud Run for the Large Dataset

To train the linear model:

```
JOB_ID="largeclicks_linear_${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud beta ml jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config-large.yaml \
  --async \
  -- \
  --dataset large \
  --model_type linear \
  --l2_regularization 500 \
  --eval_steps 1000 \
  --output_path "${GCS_PATH}/output/${JOB_ID}" \
  --metadata_path "${GCS_PATH}/preproc/metadata.json" \
  --eval_data_paths "${GCS_PATH}/preproc/features_eval*" \
  --train_data_paths "${GCS_PATH}/preproc/features_train*"
```

To train the deep model:

```
JOB_ID="largeclicks_deep_${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud beta ml jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config-large.yaml \
  --async \
  -- \
  --dataset large \
  --model_type deep \
  --hidden_units 1062 1062 1062 1062 1062 1062 1062 1062 1062 1062 1062 \
  --batch_size 512 \
  --eval_steps 1000 \
  --output_path "${GCS_PATH}/output/${JOB_ID}" \
  --metadata_path "${GCS_PATH}/preproc/metadata.json" \
  --eval_data_paths "${GCS_PATH}/preproc/features_eval*" \
  --train_data_paths "${GCS_PATH}/preproc/features_train*"
```
