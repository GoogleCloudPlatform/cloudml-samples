# Census: TensorFlow Vanilla Sample

This sample uses the [TensorFlow](https://tensorflow.org) low level APIs and
[Google Cloud Machine Learning Engine](https://cloud.google.com/ml) to demonstrate
the single node and distributed TF vanilla version for Census Income Dataset.
Setup instructions for the [Cloud ML Engine](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

## Download the data
Follow the [Census Income
Dataset](https://www.tensorflow.org/tutorials/wide/#reading_the_census_data) link to download the data. You can also download directly from [here](https://archive.ics.uci.edu/ml/datasets/Census+Income).

 * Training file is `adult.data`
 * Evaluation file is `adult.test`


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
 * Install [Pandas](http://pandas.pydata.org/pandas-docs/stable/install.html#installing-from-pypi)
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
python trainer/task.py --train_data_path $TRAIN_DATA_PATH \
                       --eval_data_path $EVAL_DATA_PATH \
                       [--max_steps $MAX_STEPS]
```

### Using gcloud local
Run the code on your local machine using `gcloud`. This allows you to mock
running it on the cloud:

```
gcloud beta ml local train --package-path trainer \
                           --module-name trainer.task \
                           -- \
                           --train_data_path $TRAIN_DATA_PATH \
                           --eval_data_path $EVAL_DATA_PATH \
                           [--max_steps $MAX_STEPS]
```

### Using Cloud ML Engine
Run the code on Cloud ML Engine using `gcloud`:

```
gcloud beta ml jobs submit training $JOB_NAME \
                                    --job-dir $GCS_LOCATION_OUTPUT \
                                    --runtime-version 1.0 \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train_data_path $TRAIN_GCS_FILE \
                                    --eval_data_path $EVAL_GCS_FILE \
                                    --max_steps 200
```
## Accuracy
You should see an accuracy of around `82.84%` for default number of training steps.

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
                           --job_dir $JOB_DIR \
                           --distributed True
```

### Using Cloud ML Engine
Run the distributed training code on cloud using `gcloud`.

```
gcloud beta ml jobs submit training $JOB_NAME \
                                    --job-dir $JOB_DIR \
                                    --scale-tier $SCALE_TIER \
                                    --runtime-version 1.0 \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train_data_path $GCS_TRAIN_PATH \
                                    --eval_data_path $GCS_EVAL_PATH \
                                    --max_steps $MAX_STEPS \
                                    --distributed True
```
