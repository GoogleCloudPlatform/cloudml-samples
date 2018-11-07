# Iris Dataset

Multi-class Classification

- - -

The is an example of a Multi-class problem used to classify a set of irises in three different classes. 
The following four numerical features describe their geometrical shape:

 - Sepal Length
 - Sepal Width
 - Petal Length
 - Petal Width

This sample uses TensorFlow:

* The sample provided in the [estimator](/estimator) folder uses the high level
  `tf.contrib.learn.Estimator` API. This API is great for fast iteration, and
  quickly adapting models to your own datasets without major code overhauls.

All the models provided in this directory can be run on the Cloud Machine Learning Engine. To follow along, check out 
the setup instructions [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

## Dataset
The [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) that this sample
uses for training is hosted by the [UC Irvine Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/). We have also hosted the data
on Google Cloud Storage:

 * Training file is [`iris_training.csv`](https://storage.googleapis.com/cloud-samples-data/ml-engine/iris/iris_training.csv)
 * Evaluation file is [`iris_test.csv`](https://storage.googleapis.com/cloud-samples-data/ml-engine/iris/iris_test.csv)

### Disclaimer
This dataset is provided by a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.

### Setup instructions 

In order to satisfy Cloud ML Engine project structure requirements. The basic project structure will look something like this:

```shell
.
├── README.md
├── setup.py
└── trainer
    ├── __init__.py
    ├── model.py
    └── task.py
```

#### Project configuration file: `setup.py`

The `setup.py` file is run on the Cloud ML Engine server to install packages/dependencies and set a few options.

Technically, Cloud ML Engine [requires a TensorFlow application to be pre-packaged] so that it can install it on the 
servers it spins up. However, if you supply a `setup.py` in the project root directory, then Cloud ML Engine will
create the package for you.


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

 * Install the Python dependencies. `pip install --upgrade -r requirements.txt`

### Get your training data

We host the Iris files required for you to run this sample. Files are hosted in a Public Bucket.

## Run the model locally

Run the code on your local machine:

Please run the export and copy statements first:

```
IRIS_DATA=data
mkdir $IRIS_DATA
gsutil cp gs://cloud-samples-data/ml-engine/iris/iris_training.csv $IRIS_DATA
gsutil cp gs://cloud-samples-data/ml-engine/iris/iris_test.csv $IRIS_DATA
```

Define variables:

```
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=iris_$DATE
export TRAIN_FILE=$IRIS_DATA/iris_training.csv
export EVAL_FILE=$IRIS_DATA/iris_test.csv
export TRAIN_STEPS=1000
```

Run the model with python (local)

```
python -m trainer.task --train-file $TRAIN_FILE \
                       --eval-file $EVAL_FILE \
                       --job-dir $OUTPUT_DIR \
                       --train-steps $TRAIN_STEPS \
                       --eval-steps 100
```

## Training using gcloud local

Run the code on your local machine using `gcloud`. This allows you to "mock"
running it on the Google Cloud:

```
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=iris_$DATE
rm -rf $OUTPUT_DIR
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/iris/iris_training.csv
export EVAL_FILE=gs://cloud-samples-data/ml-engine/iris/iris_test.csv
export TRAIN_STEPS=1000
export EVAL_STEPS=100
```

```
gcloud ml-engine local train --package-path trainer \
                           --module-name trainer.task \
                           -- \
                           --train-file $TRAIN_FILE \
                           --eval-file $EVAL_FILE \
                           --job-dir $OUTPUT_DIR \
                           --train-steps $TRAIN_STEPS \
                           --eval-steps $EVAL_STEPS
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
export BUCKET_NAME=your-bucket-name
export JOB_NAME=iris_$DATE
export OUTPUT_DIR=gs://$BUCKET_NAME/models/iris/$JOB_NAME
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/iris/iris_training.csv
export EVAL_FILE=gs://cloud-samples-data/ml-engine/iris/iris_test.csv
export TRAIN_STEPS=1000
export EVAL_STEPS=100
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.10 \
                                    --job-dir $OUTPUT_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-file $TRAIN_FILE \
                                    --eval-file $EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps $EVAL_STEPS
```

## TensorBoard
Run the TensorBoard to inspect the details about the graph.

```
tensorboard --logdir=$OUTPUT_DIR
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
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=iris_$DATE
rm -rf $OUTPUT_DIR
export TRAIN_STEPS=1000
export EVAL_STEPS=100
```

```
gcloud ml-engine local train --package-path trainer \
                           --module-name trainer.task \
                           --distributed \
                           -- \
                           --train-file $TRAIN_FILE \
                           --eval-file $EVAL_FILE \
                           --train-steps $TRAIN_STEPS \
                           --eval-steps $EVAL_STEPS \
                           --job-dir $OUTPUT_DIR

```

### Using Cloud ML Engine
Run the distributed training code on cloud using `gcloud`.

```
export BUCKET_NAME=your-bucket-name
export SCALE_TIER=STANDARD_1
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_$DATE
export OUTPUT_DIR=gs://$BUCKET_NAME/models/iris/$JOB_NAME
export TRAIN_STEPS=1000
export EVAL_STEPS=100
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --scale-tier $SCALE_TIER \
                                    --runtime-version 1.10 \
                                    --job-dir $OUTPUT_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-file $TRAIN_FILE \
                                    --eval-file $EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps $EVAL_STEPS
```

# Hyperparameter Tuning
Cloud ML Engine allows you to perform Hyperparameter tuning to find out the
most optimal hyperparameters. See [Overview of Hyperparameter Tuning]
(https://cloud.google.com/ml/docs/concepts/hyperparameter-tuning-overview) for more details.

## Running Hyperparameter Job

Running Hyperparameter job is almost exactly same as Training job except that
you need to add the `--config` argument.

```
export BUCKET_NAME=your-bucket-name
export SCALE_TIER=STANDARD_1
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_$DATE
export HPTUNING_CONFIG=hptuning_config.yaml
export OUTPUT_DIR=gs://$BUCKET_NAME/models/iris/$JOB_NAME
export TRAIN_STEPS=1000
export EVAL_STEPS=100
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --scale-tier $SCALE_TIER \
                                    --runtime-version 1.10 \
                                    --config $HPTUNING_CONFIG \
                                    --job-dir $OUTPUT_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-file $TRAIN_FILE \
                                    --eval-file $EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps $EVAL_STEPS

```

You can run the TensorBoard command to see the results of different runs and
compare accuracy / auroc numbers:

```
tensorboard --logdir=$OUTPUT_DIR
```

## Run Predictions

### Create A Prediction Service

Once your training job has finished, you can use the exported model to create a prediction server. To do this you first create a model:

```
gcloud ml-engine models create iris --regions us-central1
```

Then we'll look up the exact path that your exported trained model binaries live in:

```
gsutil ls -r $OUTPUT_DIR/export
```


 * Estimator Based: You should see a directory named `$OUTPUT_DIR/export/exporter/<timestamp>`.
```
export MODEL_BINARIES=$OUTPUT_DIR/export/exporter/<timestamp>
```

 * Low Level Based: You should see a directory named `$OUTPUT_DIR/export/JSON/`
   for `JSON`. See other formats `CSV` and `TFRECORD`.
 
```
export MODEL_BINARIES=$OUTPUT_DIR/export/CSV/
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
    --output-path $OUTPUT_DIR/predictions
```

Check the status of your prediction job:

```
gcloud ml-engine jobs describe $JOB_NAME
```

Once the job is `SUCCEEDED` you can check the results in `--output-path`.


### Disclaimer

This dataset is provided by a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.