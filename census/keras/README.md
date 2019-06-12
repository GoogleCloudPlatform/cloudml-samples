### Predicting income with the Census Income Dataset using Keras

This is the Open Source Keras version of the Census sample. The sample runs both as a
standalone Keras code and on AI Platform.

## Download the data
The [Census Income Data
Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this sample
uses for training is hosted by the [UC Irvine Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/). We have hosted the data
on Google Cloud Storage in a slightly cleaned form:

 * Training file is `adult.data.csv`
 * Evaluation file is `adult.test.csv`

```
TRAIN_FILE=adult.data.csv
EVAL_FILE=adult.test.csv

GCS_TRAIN_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv
GCS_EVAL_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.test.csv

gsutil cp $GCS_TRAIN_FILE $TRAIN_FILE
gsutil cp $GCS_EVAL_FILE $EVAL_FILE
```

## Virtual environment

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:

 * Install [Virtual](https://virtualenv.pypa.io/en/stable/) env
   * Create virtual environment `virtualenv census_keras`
   * Activate env `source census_keras/bin/activate`
 * Install [Miniconda](https://conda.io/miniconda.html)
   * Create conda environment `conda create --name census_keras python=2.7`
   * Activate env `source activate census_keras`


## Install dependencies

 * Install [gcloud](https://cloud.google.com/sdk/gcloud/)
 * Install the python dependencies. `pip install --upgrade -r requirements.txt`

## Using local python

You can run the Keras code locally

```
JOB_DIR=census_keras
TRAIN_STEPS=2000
python -m trainer.task --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --job-dir $JOB_DIR \
                       --train-steps $TRAIN_STEPS
```

## Training using gcloud local

You can run Keras training using gcloud locally

```
JOB_DIR=census_keras
TRAIN_STEPS=200
gcloud ai-platform local train --package-path trainer \
                             --module-name trainer.task \
                             -- \
                             --train-files $TRAIN_FILE \
                             --eval-files $EVAL_FILE \
                             --job-dir $JOB_DIR \
                             --train-steps $TRAIN_STEPS
```

## Prediction using gcloud local

You can run prediction on the SavedModel created from Keras HDF5 model

```
python preprocess.py sample.json
```

```
gcloud ai-platform local predict --model-dir=$JOB_DIR/export \
                               --json-instances sample.json
```

## Training using AI Platform

You can train the model on AI Platform

```
gcloud ai-platform jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.4 \
                                    --job-dir $JOB_DIR \
                                    --package-path trainer \
                                    --module-name trainer.task \
                                    --region us-central1 \
                                    -- \
                                    --train-files $GCS_TRAIN_FILE \
                                    --eval-files $GCS_EVAL_FILE \
                                    --train-steps $TRAIN_STEPS
```

## Prediction using AI Platform

You can perform prediction on AI Platform by following the steps below.
Create a model on AI Platform

```
gcloud ai-platform models create keras_model --regions us-central1
```

Export the model binaries

```
MODEL_BINARIES=$JOB_DIR/export
```

Deploy the model to the prediction service

```
gcloud ai-platform versions create v1 --model keras_model --origin $MODEL_BINARIES --runtime-version 1.2
```

Create a processed sample from the data

```
python preprocess.py sample.json

```

Run the online prediction

```
gcloud ai-platform predict --model keras_model --version v1 --json-instances sample.json
```
