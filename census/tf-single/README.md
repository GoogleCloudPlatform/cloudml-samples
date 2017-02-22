# Census: TensorFlow Vanilla Sample

This sample uses the [TensorFlow](https://tensorflow.org) low level APIs and
[Google Cloud Machine Learning API](https://cloud.google.com/ml) to demonstrate
the single node and distributed TF vanilla version for Census Income Dataset.

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
 * Install [TensorFlow](https://www.tensorflow.org/install/)
 * Install [Pandas](http://pandas.pydata.org/pandas-docs/stable/install.html#installing-from-pypi)


# Single Node Version
Single node version runs TF code on a single instance. You can run the exact
same code locally and on Cloud ML.

## How to run the code
You can run the code either as a stand-alone python program or using `gcloud`.
See options below:

### Local Run
Run the code on your local machine:

```
python trainer/task.py --train_data_path $TRAIN_DATA_PATH \
                       --eval_data_path $EVAL_DATA_PATH \
                       [--max_steps $MAX_STEPS]
```

### Gcloud Local Run
Run the code on your local machine using `gcloud`:

```
gcloud beta ml local train --package-path trainer \
                           --module-name trainer.task \
                           -- \
                           --train_data_path $TRAIN_DATA_PATH \
                           --eval_data_path $EVAL_DATA_PATH \
                           [--max_steps $MAX_STEPS]
```

### Gcloud Cloud ML Engine Run
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
You should see an accuracy of `82.84%` for default number of training steps.

# Distributed Version
Distributed version of the code uses Distributed TensorFlow. The main change to
make the distributed version work is usage of `TF_CONFIG` environment variable.
The environment variable is generated using `gcloud` and parsed to create a
`ClusterSpec`.

## How to run the code

### Gcloud Local Run
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

### Gcloud Cloud ML Engine Run
