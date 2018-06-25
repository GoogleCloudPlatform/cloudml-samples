# Predicting Income with the Census Income Dataset

Predict a person's income level.

- - -

The example in this directory shows how to use [tf transform](https://github.com/tensorflow/transform)
together with [Cloud Dataflow](https://cloud.google.com/dataflow) and [Cloud ML Engine](https://cloud.google.com/ml-engine/). As with the other Census example this code can be run locally or on cloud.

## Preparing the data for training
The [Census Income Data
Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this sample
uses for training is hosted by the [UC Irvine Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/). We have hosted the data
on Google Cloud Storage in a slightly cleaned form:

 * Training file is `adult.data.csv`
 * Evaluation file is `adult.test.csv`

### Disclaimer
This dataset is provided by a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.

### Set Environment Variables
Please run the export and copy statements first:

```
TRAIN_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv
EVAL_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.test.csv
```

### Use local training files.

```
mkdir census_data

gsutil cp $TRAIN_FILE census_data/adult.data.csv
gsutil cp $EVAL_FILE census_data/adult.test.csv

TRAIN_FILE_LOCAL=census_data/adult.data.csv
EVAL_FILE_LOCAL=census_data/adult.test.csv
```


## Virtual environment
Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:
 * Install [Virtual](https://virtualenv.pypa.io/en/stable/) env
   * Create virtual environment `virtualenv single-tf`
   * Activate env `source single-tf/bin/activate`
 * Install [Miniconda](https://conda.io/miniconda.html)
   * Create conda environment `conda create --name single-tf python=2.7`
   * Activate env `source activate single-tf`


## Install dependencies

 * Install [gcloud](https://cloud.google.com/sdk/gcloud/)
 * Install the python dependencies. `pip install --upgrade -r requirements.txt`

# Preprocessing with tf transform
## Running locally

Run the code on your local machine:

```
export TFT_OUTPUT_DIR_LOCAL=census_data_tft
rm -rf $TFT_OUTPUT_DIR_LOCAL

python preprocess.py --train-data-file $TRAIN_FILE_LOCAL \
                     --test-data-file $EVAL_FILE_LOCAL \
                     --root-train-data-out train \
                     --root-test-data-out test \
                     --working-dir $TFT_OUTPUT_DIR_LOCAL \
                     --temp_location ${TFT_OUTPUT_DIR_LOCAL}/tmp

```

## Preprocessing with Cloud Dataflow

```
gcloud auth application-default login

export TFT_OUTPUT_DIR=gs://<my-bucket>/path/to/my/jobs/
export REGION=<your-gcp-region>
export PROJECT=<your-gcp-project>
export JOB_NAME=census-tft

python preprocess.py --train-data-file $TRAIN_FILE \
                     --test-data-file $EVAL_FILE \
                     --root-train-data-out train \
                     --root-test-data-out test \
                     --working-dir $TFT_OUTPUT_DIR \
                     --runner DataflowRunner \
                     --requirements_file requirements_dataflow.txt \
                     --region $REGION \
                     --project $PROJECT \
                     --job_name $JOB_NAME
```


# Single Node Training
Single node training runs TensorFlow code on a single instance. You can run the exact
same code locally and on Cloud ML Engine.

## How to run the code
You can run the code either as a stand-alone python program or using `gcloud`.
See options below:

### Using local python
Run the code on your local machine:

```
export TRAIN_STEPS=1000
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=census_$DATE
rm -rf $OUTPUT_DIR
```

### Model location reuse
Its worth calling out that unless you want to restore training from the
[checkpoints](https://www.tensorflow.org/get_started/checkpoints) saved
in the old model output dir, model location should be a new location.

```
python -m trainer.task --tft-working-dir $TFT_OUTPUT_DIR_LOCAL \
                       --train-filebase train \
                       --eval-filebase test \
                       --train-steps $TRAIN_STEPS \
                       --job-dir $OUTPUT_DIR

```

### Using gcloud local
Run the code on your local machine using `gcloud`. This allows you to mock
running it on the cloud:

```
export TRAIN_STEPS=1000
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=census_$DATE
rm -rf $OUTPUT_DIR
```

```
gcloud ml-engine local train --module-name trainer.task \
                             --package-path trainer \
                             -- \
                             --tft-working-dir $TFT_OUTPUT_DIR_LOCAL \
                             --train-filebase train \
                             --eval-filebase test \
                             --train-steps $TRAIN_STEPS \
                             --job-dir $OUTPUT_DIR

```

### Using Cloud ML Engine
Run the code on Cloud ML Engine using `gcloud`. Note how `--job-dir` comes
before `--` while training on the cloud and this is so that we can have
different trial runs during Hyperparameter tuning. See more information about
[training job arguments](https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs#submitting_the_job).

```
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=census_$DATE
export GCS_JOB_DIR=gs://<my-bucket>/path/to/my/jobs/$JOB_NAME
echo $GCS_JOB_DIR
export TRAIN_STEPS=5000
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.4 \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer \
                                    --region $REGION \
                                    --project $PROJECT \
                                    -- \
                                    --tft-working-dir $TFT_OUTPUT_DIR \
                                    --train-filebase train \
                                    --eval-filebase test \
                                    --train-steps $TRAIN_STEPS

```

## Tensorboard
Run the Tensorboard to inspect the details about the graph.

```
tensorboard --logdir=$GCS_JOB_DIR
```

## Accuracy and Output
You should see the output for default number of training steps and approx accuracy close to `80%`.

# Distributed Node Training
Distributed node training uses [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).
The main change to make the distributed version work is usage of [TF_CONFIG](https://cloud.google.com/ml-engine/docs/tensorflow/distributed-training-details)
environment variable. The environment variable is generated using `gcloud` and parsed to create a
[ClusterSpec](https://www.tensorflow.org/deploy/distributed#create_a_tftrainclusterspec_to_describe_the_cluster). See the [ScaleTier](https://cloud.google.com/ml-engine/docs/pricing#scale-tier) for predefined tiers.

## How to run the code
You can run the code either locally or on cloud using `gcloud`.

### Using gcloud local
Run the distributed training code locally using `gcloud`.

```
export TRAIN_STEPS=1000
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=census_distributed_$DATE
rm -rf $OUTPUT_DIR
```

```
gcloud ml-engine local train --module-name trainer.task \
                             --package-path trainer \
                             --distributed \
                             -- \
                             --tft-working-dir $TFT_OUTPUT_DIR_LOCAL \
                             --train-filebase train \
                             --eval-filebase test \
                             --train-steps $TRAIN_STEPS \
                             --job-dir $OUTPUT_DIR

```

### Using Cloud ML Engine
Run the distributed training code on cloud using `gcloud`.

```
export SCALE_TIER=STANDARD_1
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=census_${SCALE_TIER}_$DATE
export GCS_JOB_DIR=gs://<my-bucket>/path/to/my/models/$JOB_NAME
echo $GCS_JOB_DIR
export TRAIN_STEPS=5000
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.4 \
                                    --scale-tier $SCALE_TIER \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer \
                                    --region $REGION \
                                    --project $PROJECT \
                                    -- \
                                    --tft-working-dir $TFT_OUTPUT_DIR \
                                    --train-filebase train \
                                    --eval-filebase test \
                                    --train-steps $TRAIN_STEPS

```

# Hyperparameter Tuning
Cloud ML Engine allows you to perform Hyperparameter tuning to find out the
most optimal hyperparameters. See [Overview of Hyperparameter Tuning](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview)
for more details.

## Running Hyperparameter Job

Running Hyperparameter job is almost exactly same as Training job except that
you need to add the `--config` argument.

```
export HPTUNING_CONFIG=../hptuning_config.yaml
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=census_hptuning_$DATE
export GCS_JOB_DIR=gs://<my-bucket>/path/to/my/models/$JOB_NAME
echo $GCS_JOB_DIR
export TRAIN_STEPS=5000

```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.4 \
                                    --config $HPTUNING_CONFIG \
                                    --scale-tier $SCALE_TIER \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer \
                                    --region $REGION \
                                    --project $PROJECT \
                                    -- \
                                    --tft-working-dir $TFT_OUTPUT_DIR \
                                    --train-filebase train \
                                    --eval-filebase test \
                                    --train-steps $TRAIN_STEPS

```

You can run the Tensorboard command to see the results of different runs and
compare accuracy / auroc numbers:

```
tensorboard --logdir=$GCS_JOB_DIR
```

## Run Predictions

### Create A Prediction Service

Once your training job has finished, you can use the exported model to create a prediction server. To do this you first create a model:

```
gcloud ml-engine models create census --regions $REGION
```

Then we'll look up the exact path that your exported trained model binaries live in:

```
gsutil ls -r $GCS_JOB_DIR/export
```


 * You should see a directory named `$GCS_JOB_DIR/export/tft_classifier/<timestamp>`.
 ```
 export MODEL_BINARIES=$GCS_JOB_DIR/export/tft_classifier/<timestamp>
 ```

```
gcloud ml-engine versions create v1 --model census --origin $MODEL_BINARIES --runtime-version 1.4
```

### (Optional) Inspect the model binaries with the SavedModel CLI
TensorFlow ships with a CLI that allows you to inspect the signature of exported binary files. To do this run:

```
saved_model_cli show --dir $MODEL_BINARIES --tag serve --signature_def predict
```

### Run Online Predictions

You can now send prediction requests to the API. To test this out you can use the `gcloud ml-engine predict` tool:

```
gcloud ml-engine predict --model census --version v1 --json-instances ../test.json
```

You should see a response with the predicted labels of the examples!
