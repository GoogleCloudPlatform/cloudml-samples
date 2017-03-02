# Census: TensorFlow Low-Level API Sample

This sample uses the [TensorFlow](https://tensorflow.org) low level APIs and
[Google Cloud Machine Learning Engine](https://cloud.google.com/ml) to demonstrate
the single node and distributed TF vanilla version for Census Income Dataset.
Setup instructions for the [Cloud ML Engine](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

## Download the data
The [Census Income Data
Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this sample
uses for training is hosted by the [UC Irvine Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/). We have hosted the data
on Google Cloud Storage:

 * Training file is `adult.data.csv`
 * Evaluation file is `adult.test.csv`

### Disclaimer
The source of this dataset is from a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.

```
export CENSUS_DATA=census_data
export TRAIN_FILE=adult.data.csv
export EVAL_FILE=adult.test.csv
mkdir $CENSUS_DATA

export TRAIN_GCS_FILE=gs://cloudml-public/census/data/$TRAIN_FILE
export EVAL_GCS_FILE=gs://cloudml-public/census/data/$EVAL_FILE

gsutil cp $TRAIN_GCS_FILE $CENSUS_DATA
gsutil cp $EVAL_GCS_FILE $CENSUS_DATA
```


## Virtual environment
There are two options for the virtual environments:
 * Install [Virtual](https://virtualenv.pypa.io/en/stable/) env
   * Create virtual environment `virtualenv single-tf`
   * Activate env `source single-tf/bin/activate`
 * Install [Miniconda](https://conda.io/miniconda.html)
   * Create conda environment `conda create --name single-tf python=2.7`
   * Activate env `source activate single-tf`


## Install dependencies
Install the following dependencies:
 * Install [Cloud SDK](https://cloud.google.com/sdk/)
 * Install [TensorFlow](https://www.tensorflow.org/install/)
 * Install [gcloud](https://cloud.google.com/sdk/gcloud/)


# Single Node Training
Single node training runs TensorFlow code on a single instance. You can run the exact
same code locally and on Cloud ML Engine.

## How to run the code
You can run the code either as a stand-alone python program or using `gcloud`.
See options below:

### Using local python
Run the code on your local machine:

```
export OUTPUT_DIR=census_output
python trainer/task.py --train_data_path $CENSUS_DATA/$TRAIN_FILE \
                       --eval_data_path $CENSUS_DATA/$EVAL_FILE \
                       --output_dir $OUTPUT_DIR
                       [--max_steps $MAX_STEPS]
```

### Using gcloud local
Run the code on your local machine using `gcloud`. This allows you to mock
running it on the cloud:

```
gcloud beta ml local train --package-path trainer \
                           --module-name trainer.task \
                           -- \
                           --train_data_path $CENSUS_DATA/$TRAIN_FILE \
                           --eval_data_path $CENSUS_DATA/$EVAL_FILE \
                           --output_dir $OUTPUT_DIR
```

### Using Cloud ML Engine
Run the code on Cloud ML Engine using `gcloud`:

```
gcloud beta ml jobs submit training $JOB_NAME \
                                    --runtime-version 1.0 \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train_data_path $TRAIN_GCS_FILE \
                                    --eval_data_path $EVAL_GCS_FILE \
                                    --output_dir $GCS_OUTPUT_DIR \
                                    --max_steps $MAX_STEPS
```
## Accuracy and Output
You should see the output for default number of training steps and approx accuracy close to `80.25%`.

```
INFO:tensorflow:global_step/sec: 280.197
Accuracy at step: 101 is 72.25%
INFO:tensorflow:global_step/sec: 621.539
Accuracy at step: 202 is 77.75%
INFO:tensorflow:global_step/sec: 621.029
Accuracy at step: 305 is 78.00%
INFO:tensorflow:global_step/sec: 662.139
Accuracy at step: 407 is 78.75%
INFO:tensorflow:global_step/sec: 640.948
Accuracy at step: 508 is 75.50%
INFO:tensorflow:global_step/sec: 668.127
Accuracy at step: 609 is 73.75%
INFO:tensorflow:global_step/sec: 676.06
Accuracy at step: 710 is 76.00%
INFO:tensorflow:global_step/sec: 636.939
Accuracy at step: 812 is 76.25%
INFO:tensorflow:global_step/sec: 661.389
Accuracy at step: 913 is 80.25%
```


# Distributed Node Training
Distributed node training uses [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).
The main change to make the distributed version work is usage of [TF_CONFIG](https://cloud.google.com/ml/reference/configuration-data-structures#tf_config_environment_variable)
environment variable. The environment variable is generated using `gcloud` and parsed to create a
[ClusterSpec](https://www.tensorflow.org/deploy/distributed#create_a_tftrainclusterspec_to_describe_the_cluster). See the [ScaleTier](https://cloud.google.com/ml/pricing#ml_training_units_by_scale_tier) for predefined tiers

## How to run the code
You can run the code either locally or on cloud using `gcloud`.

### Using gcloud local
Run the distributed training code locally using `gcloud`.

```
gcloud beta ml local train --package-path trainer \
                           --module-name trainer.task \
                           --parameter-server-count $PS_SERVER_COUNT \
                           --worker-count $WORKER_COUNT \
                           --distributed \
                           -- \
                           --train_data_path $TRAIN_DATA_PATH \
                           --eval_data_path $EVAL_DATA_PATH \
                           --max_steps $MAX_STEPS \
                           --output_dir $OUTPUT_DIR
```

### Using Cloud ML Engine
Run the distributed training code on cloud using `gcloud`.

```
gcloud beta ml jobs submit training $JOB_NAME \
                                    --scale-tier $SCALE_TIER \
                                    --runtime-version 1.0 \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train_data_path $TRAIN_GCS_FILE \
                                    --eval_data_path $EVAL_GCS_FILE \
                                    --max_steps $MAX_STEPS \
                                    --output_dir $GCS_OUTPUT_DIR
```

# Hyperparameter Tuning
Cloud ML Engine allows you to perform Hyperparameter tuning to find out the
most optimal hyperparameters. See [Overview of Hyperparameter Tuning]
(https://cloud.google.com/ml/docs/concepts/hyperparameter-tuning-overview) for more details.

## Running Hyperparameter Job

Running Hyperparameter job is almost exactly same as Training job except that
you need to add the `--config` argument.

```
export HPTUNING_CONFIG=hptuning_config.yaml
```

```
gcloud beta ml jobs submit training $JOB_NAME \
                                    --scale-tier $SCALE_TIER \
                                    --runtime-version 1.0 \
                                    --config $HPTUNING_CONFIG
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train_data_path $TRAIN_GCS_FILE \
                                    --eval_data_path $EVAL_GCS_FILE \
                                    --max_steps $MAX_STEPS \
                                    --output_dir $GCS_OUTPUT_DIR
```
