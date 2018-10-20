# Iris Dataset

Multiclass classification

- - -

The task is to classify a set of irises in three different classes, depending on four numerical features describing their geometrical shape:

 - Sepal Length
 - Sepal Width
 - Petal Length
 - Petal Width

These samples uses Tensorflow:

* The sample provided in the [estimator](/estimator) folder uses the high level
  `tf.contrib.learn.Estimator` API. This API is great for fast iteration, and
  quickly adapting models to your own datasets without major code overhauls.
  **Version:** ```Tensorflow Version: 1.10```


## Dataset
The [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) that this sample
uses for training is hosted by the [UC Irvine Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/). We have also hosted the data
on Google Cloud Storage:

 * Training file is [`iris_training.csv`](https://storage.googleapis.com/cloud-samples-data/ml-engine/iris/iris_training.csv)
 * Evaluation file is [`iris_test.csv`](https://storage.googleapis.com/cloud-samples-data/ml-engine/iris/iris_test.csv)

### Setup instructions 

All the models provided in this directory can be run on the Cloud Machine Learning Engine. To follow along, check out the setup instructions [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

#### Set Environment Variables
Please run the export and copy statements first:

```
GCS_TRAIN_FILE=gs://cloud-samples-data/ml-engine/iris/iris_training.csv
GCS_EVAL_FILE=gs://cloud-samples-data/ml-engine/iris/iris_test.csv
```

#### \*Optional Use local training files.

Since TensorFlow handles reading from GCS, you can run all commands below using these environment variables. However, if your network is slow or unreliable, you may want to download the files for local training.

```
IRIS_DATA=data

mkdir $IRIS_DATA

TRAIN_FILE=$IRIS_DATA/iris_training.csv
EVAL_FILE=$IRIS_DATA/iris_test.csv

gsutil cp $GCS_TRAIN_FILE $TRAIN_FILE
gsutil cp $GCS_EVAL_FILE $EVAL_FILE
```


## Virtual environment

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
without changing global python packages on your system.

There are two options for the virtual environments:
 * Install [virtualenv](https://virtualenv.pypa.io/en/stable/) env
   * Create virtual environment `virtualenv iris`
   * Activate env `source iris/bin/activate`
 * Install [miniconda](https://conda.io/miniconda.html)
   * Create conda environment `conda create --name iris python=3.5`
   * Activate env `source activate iris`


## Install dependencies

 * Install [gcloud](https://cloud.google.com/sdk/gcloud/)
 * Install the Python dependencies. `pip install --upgrade -r requirements.txt`

# Single Node Training
Single node training runs TensorFlow code on a single instance. You can run the exact
same code locally and on Cloud ML Engine.

## How to run the code
You can run the code either as a stand-alone python program or using `gcloud`.
See options below:

### Local Python

Run the code on your local machine:

```
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=iris_$DATE
rm -rf $OUTPUT_DIR
export TRAIN_STEPS=1000
```

This step is suggested in order to verify your model works locally. 

```
python -m trainer.task --train-file $TRAIN_FILE \
                       --eval-file $EVAL_FILE \
                       --job-dir $OUTPUT_DIR \
                       --train-steps $TRAIN_STEPS \
                       --eval-steps 100
```

### Using gcloud local
Run the code on your local machine using `gcloud`. This allows you to "mock"
running it on the Google Cloud:

```
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=iris_$DATE
rm -rf $OUTPUT_DIR
export TRAIN_STEPS=1000
```

```
gcloud ml-engine local train --package-path trainer \
                           --module-name trainer.task \
                           -- \
                           --train-file $TRAIN_FILE \
                           --eval-file $EVAL_FILE \
                           --job-dir $OUTPUT_DIR \
                           --train-steps $TRAIN_STEPS \
                           --eval-steps 100
```

### Using Cloud ML Engine
*NOTE* If you downloaded the training files to your local file system, be sure
to reset the `TRAIN_FILE` and `EVAL_FILE` environment variables to refer to a GCS location.
Data must be in GCS for cloud-based training.

Run the code on Cloud ML Engine using `gcloud`. Note how `--job-dir` comes
before `--` while training on the cloud and this is so that we can have
different trial runs during Hyperparameter tuning.

```
DATE=`date '+%Y%m%d_%H%M%S'`
export BUCKET_NAME=<MY-BUCKET-NAME>
export JOB_NAME=iris_$DATE
export GCS_JOB_DIR=gs://$BUCKET_NAME/models/iris/$JOB_NAME
echo $GCS_JOB_DIR
export GCS_TRAIN_FILE=gs://cloud-samples-data/ml-engine/iris/iris_training.csv
export GCS_EVAL_FILE=gs://cloud-samples-data/ml-engine/iris/iris_test.csv
export TRAIN_STEPS=1000
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.10 \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-file $GCS_TRAIN_FILE \
                                    --eval-file $GCS_EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps 100
```

## Tensorboard
Run the Tensorboard to inspect the details about the graph.

```
tensorboard --logdir=$GCS_JOB_DIR
```

## Accuracy and Output
You should see the output for default number of training steps and approx accuracy close to `90%`.

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
export TRAIN_STEPS=1000
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=iris_$DATE
rm -rf $OUTPUT_DIR
```

```
gcloud ml-engine local train --package-path trainer \
                           --module-name trainer.task \
                           --distributed \
                           -- \
                           --train-file $TRAIN_FILE \
                           --eval-file $EVAL_FILE \
                           --train-steps $TRAIN_STEPS \
                           --job-dir $OUTPUT_DIR \
                           --eval-steps 100

```

### Using Cloud ML Engine
Run the distributed training code on cloud using `gcloud`.

```
export BUCKET_NAME=<MY-BUCKET-NAME>
export SCALE_TIER=STANDARD_1
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_$DATE
export GCS_JOB_DIR=gs://$BUCKET_NAME/models/iris/$JOB_NAME
echo $GCS_JOB_DIR
export TRAIN_STEPS=1000
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --scale-tier $SCALE_TIER \
                                    --runtime-version 1.10 \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-file $GCS_TRAIN_FILE \
                                    --eval-file $GCS_EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps 100
```

# Hyperparameter Tuning
Cloud ML Engine allows you to perform Hyperparameter tuning to find out the
most optimal hyperparameters. See [Overview of Hyperparameter Tuning]
(https://cloud.google.com/ml/docs/concepts/hyperparameter-tuning-overview) for more details.

## Running Hyperparameter Job

Running Hyperparameter job is almost exactly same as Training job except that
you need to add the `--config` argument.

```
export BUCKET_NAME=<MY-BUCKET-NAME>
export SCALE_TIER=STANDARD_1
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_$DATE
export HPTUNING_CONFIG=hptuning_config.yaml
export GCS_JOB_DIR=gs://$BUCKET_NAME/models/iris/$JOB_NAME
echo $GCS_JOB_DIR
export TRAIN_STEPS=1000
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --scale-tier $SCALE_TIER \
                                    --runtime-version 1.10 \
                                    --config $HPTUNING_CONFIG \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-file $TRAIN_FILE \
                                    --eval-file $EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps 100

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
gcloud ml-engine models create iris --regions us-central1
```

Then we'll look up the exact path that your exported trained model binaries live in:

```
gsutil ls -r $GCS_JOB_DIR/export
```


 * Estimator Based: You should see a directory named `$GCS_JOB_DIR/export/exporter/<timestamp>`.
```
export MODEL_BINARIES=$GCS_JOB_DIR/export/exporter/<timestamp>
```

 * Low Level Based: You should see a directory named `$GCS_JOB_DIR/export/JSON/`
   for `JSON`. See other formats `CSV` and `TFRECORD`.
 
```
export MODEL_BINARIES=$GCS_JOB_DIR/export/CSV/
```

```
gcloud ml-engine versions create v1 --model iris --origin $MODEL_BINARIES --runtime-version 1.10
```

### (Optional) Inspect the model binaries with the SavedModel CLI
TensorFlow ships with a CLI that allows you to inspect the signature of exported binary files. To do this run:

```
SIGNATURE_DEF_KEY=`saved_model_cli show --dir $MODEL_BINARIES --tag serve | grep "SignatureDef key:" | awk 'BEGIN{FS="\""}{print $2}' | head -1`
saved_model_cli show --dir $MODEL_BINARIES --tag serve --signature_def $SIGNATURE_DEF_KEY
```

### Run Online Predictions

You can now send prediction requests to the API. To test this out you can use the `gcloud ml-engine predict` tool:

```
gcloud ml-engine predict --model iris --version v1 --json-instances test.json
```

Using CSV

```
gcloud ml-engine predict --model iris --version v1 --text-instances test.csv
```

Example:

```
6.4, 3.2, 4.5, 1.5
```

You should see a response with the predicted labels of the examples!

### Run Batch Prediction

If you have large amounts of data, and no latency requirements on receiving prediction results, you can submit a prediction job to the API. This uses the same format as online prediction, but requires data be stored in Google Cloud Storage

```
export JOB_NAME=iris_prediction
```

```
gcloud ml-engine jobs submit prediction $JOB_NAME \
    --model iris \
    --version v1 \
    --data-format TEXT \
    --region us-central1 \
    --runtime-version 1.10 \
    --input-paths gs://cloud-samples-data/ml-engine/testdata/prediction/iris.json \
    --output-path $GCS_JOB_DIR/predictions
```

Check the status of your prediction job:

```
gcloud ml-engine jobs describe $JOB_NAME
```

Once the job is `SUCCEEDED` you can check the results in `--output-path`.


### Disclaimer

This dataset is provided by a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.