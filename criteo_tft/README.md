## Criteo Sample

The Criteo sample demonstrates the capability of both linear and deep models on
the [criteo dataset](https://www.kaggle.com/c/criteo-display-ad-challenge).

## Prerequisites

*   Make sure you follow the Google Cloud ML setup
    [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up) before
    trying the sample. More documentation about Cloud ML is available
    [here](https://cloud.google.com/ml/docs/).
*   Make sure your Google Cloud project has sufficient quota.

## Install Dependencies

Install dependencies by running `pip install -r requirements.txt`

## Sample Overview

This sample consists of two parts:

### Data Pre-Processing

Data pre-processing step involves taking the TSV data as input and converting it
to [TFRecords](https://www.tensorflow.org/api_guides/python/python_io) format.

### Model Training

Model training step involves taking the pre-processed TFRecords data and
training a linear classifier using Stochastic Dual Coordinate Ascent (SDCA)
optimizer, or a deep neural network classifier.

## Criteo Dataset

The criteo dataset is available in two different sizes:

### Kaggle Challenge Dataset

Kaggle challenge dataset can be downloaded
[here](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
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

For the small dataset, we first split the `train.txt` file into training and
evaluation sets. The large dataset consists of 24 files named `day_0.txt`
through `day_23.txt`. We use the first 23 files as training data. The last file
is used for evaluation, and we rename it to `eval_day_23.txt` for easier file
matching using wildcards.

### Local Run

We recommend using local preprocessing only for testing on a small subset of the
data. You can run it as:

```
LOCAL_DATA_DIR=[download location]
head -10000 $LOCAL_DATA_DIR/train.txt > $LOCAL_DATA_DIR/train-10k.txt
tail -2000 $LOCAL_DATA_DIR/train.txt > $LOCAL_DATA_DIR/eval-2k.txt
python preprocess.py --training_data $LOCAL_DATA_DIR/train-10k.txt \
                     --eval_data $LOCAL_DATA_DIR/eval-2k.txt \
                     --output_dir $LOCAL_DATA_DIR/preproc
```

### Cloud Run

In order to run pre-processing on the Cloud run the commands below.

```
PROJECT=$(gcloud config list project --format "value(core.project)")
BUCKET="gs://${PROJECT}-ml"
```


```
# Small dataset
GCS_PATH_SMALL="${BUCKET}/${USER}/smallclicks"
head -40800000 $LOCAL_DATA_DIR/train.txt > $LOCAL_DATA_DIR/train-40m.txt
tail -5000000 $LOCAL_DATA_DIR/train.txt > $LOCAL_DATA_DIR/eval-5m.txt
gsutil -m cp $LOCAL_DATA_DIR/train-40m.txt $LOCAL_DATA_DIR/eval-5m.txt $GCS_PATH_SMALL

PREPROCESS_OUTPUT_SMALL="${GCS_PATH_SMALL}/criteo_$(date +%Y%m%d_%H%M%S)"
python preprocess.py --training_data "${GCS_PATH_SMALL}/train-40m.txt" \
                     --eval_data "${GCS_PATH_SMALL}/eval-5m.txt" \
                     --output_dir "${PREPROCESS_OUTPUT_SMALL}" \
                     --project_id "${PROJECT}" \
                     --cloud
```

```
# Large dataset
GCS_PATH_LARGE="${BUCKET}/${USER}/largeclicks"
gsutil mv $GCS_PATH_LARGE/day_23.txt $GCS_PATH_LARGE/eval_day_23.txt

PREPROCESS_OUTPUT_LARGE="${GCS_PATH_LARGE}/criteo_$(date +%Y%m%d_%H%M%S)"
python preprocess.py --training_data "${GCS_PATH_LARGE}/day_*" \
                     --eval_data "${GCS_PATH_LARGE}/eval_day_*" \
                     --output_dir "${PREPROCESS_OUTPUT_LARGE}" \
                     --project_id "${PROJECT}" \
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
          --raw_metadata_path $LOCAL_OUTPUT_DIR/raw_metadata \
          --transformed_metadata_path $LOCAL_OUTPUT_DIR/transformed_metadata \
          --transform_savedmodel $LOCAL_OUTPUT_DIR/transform_fn \
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
          --raw_metadata_path $LOCAL_OUTPUT_DIR/raw_metadata \
          --transformed_metadata_path $LOCAL_OUTPUT_DIR/transformed_metadata \
          --transform_savedmodel $LOCAL_OUTPUT_DIR/transform_fn \
          --output_path $TRAINING_OUTPUT_PATH
```

Running time varies depending on your machine. Typically the linear model takes
at least 2 hours to train, and the deep model more than 8 hours. You can use
[Tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/) to
follow the job's progress.

### Cloud Run for the Small Dataset

You can train using either a single worker (config-single.yaml), or using
multiple workers and parameter servers (config-small.yaml).

To train the linear model:

```
JOB_ID="smallclicks_linear_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config-small.yaml \
  --async \
  -- \
  --dataset kaggle \
  --model_type linear \
  --l2_regularization 100 \
  --output_path "${GCS_PATH_SMALL}/model/${JOB_ID}" \
  --raw_metadata_path "${PREPROCESS_OUTPUT_SMALL}/raw_metadata" \
  --transformed_metadata_path "${PREPROCESS_OUTPUT_SMALL}/transformed_metadata" \
  --transform_savedmodel "${PREPROCESS_OUTPUT_SMALL}/transform_fn" \
  --eval_data_paths "${PREPROCESS_OUTPUT_SMALL}/features_eval*" \
  --train_data_paths "${PREPROCESS_OUTPUT_SMALL}/features_train*"
```

To train the deep model:

```
JOB_ID="smallclicks_deep_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
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
  --output_path "${GCS_PATH_SMALL}/model/${JOB_ID}" \
  --raw_metadata_path "${PREPROCESS_OUTPUT_SMALL}/raw_metadata" \
  --transformed_metadata_path "${PREPROCESS_OUTPUT_SMALL}/transformed_metadata" \
  --transform_savedmodel "${PREPROCESS_OUTPUT_SMALL}/transform_fn" \
  --eval_data_paths "${PREPROCESS_OUTPUT_SMALL}/features_eval*" \
  --train_data_paths "${PREPROCESS_OUTPUT_SMALL}/features_train*"
```

When using the [distributed configuration](config-small.yaml), the linear model
may take as little as 10 minutes to train, and the deep model should finish in
around 90 minutes. Again you can point Tensorboard to the output path to follow
training progress.

### Cloud Run for the Large Dataset

To train the linear model:

```
JOB_ID="largeclicks_linear_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config-large.yaml \
  --async \
  -- \
  --dataset large \
  --model_type linear \
  --l2_regularization 3000 \
  --eval_steps 1000 \
  --output_path "${GCS_PATH_LARGE}/model/${JOB_ID}" \
  --raw_metadata_path "${PREPROCESS_OUTPUT_LARGE}/raw_metadata" \
  --transformed_metadata_path "${PREPROCESS_OUTPUT_LARGE}/transformed_metadata" \
  --transform_savedmodel "${PREPROCESS_OUTPUT_LARGE}/transform_fn" \
  --eval_data_paths "${PREPROCESS_OUTPUT_LARGE}/features_eval*" \
  --train_data_paths "${PREPROCESS_OUTPUT_LARGE}/features_train*"
```

To train the linear model without crosses, add the option `--ignore_crosses` and
use `--l2_regularization 1000` for best results.

To train the deep model:

```
JOB_ID="largeclicks_deep_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --config config-large.yaml \
  --async \
  -- \
  --dataset large \
  --model_type deep \
  --hidden_units 1024 512 256 \
  --batch_size 512 \
  --eval_steps 250 \
  --output_path "${GCS_PATH_LARGE}/model/${JOB_ID}" \
  --raw_metadata_path "${PREPROCESS_OUTPUT_LARGE}/raw_metadata" \
  --transformed_metadata_path "${PREPROCESS_OUTPUT_LARGE}/transformed_metadata" \
  --transform_savedmodel "${PREPROCESS_OUTPUT_LARGE}/transform_fn" \
  --eval_data_paths "${PREPROCESS_OUTPUT_LARGE}/features_eval*" \
  --train_data_paths "${PREPROCESS_OUTPUT_LARGE}/features_train*"
```
