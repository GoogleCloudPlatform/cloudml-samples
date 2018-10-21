# Predicting Income with the Census Income Dataset

Predict a person's income level.

- - -

Using census data which contains data a person's age, education, marital status, 
and occupation (the features), we will try to predict whether or not the person earns 
more than 50,000 dollars a year (the target label). We will train a logistic regression 
model that, given an individual's information, outputs a number between 0 and 1—this can 
be interpreted as the probability that the individual has an annual income of over 50,000 dollars.

As a modeler and developer, think about how this data is used and the potential benefits and harm a model's predictions can cause. A model like this could reinforce societal biases and disparities. Is each feature relevant to the problem you want to solve or will it introduce bias? For more information, read about [ML fairness](https://developers.google.com/machine-learning/fairness-overview/).

## Examples

There are five samples provided in this directory. They allow you to move from
single-worker training to distributed training without any code changes, and
make it easy to export model binaries for prediction, but with the following
distiction:
  
* The sample provided in [TensorFlow Core](./tensorflowcore) uses the low level
  bindings to build a model. This example is great for understanding the
  underlying workings of TensorFlow, best practices when using the low-level
  APIs.

* The sample provided in [Custom Estimator](./customestimator) uses the custom
  Tensorflow `tf.estimator.EstimatorSpec` to create a High level custom Estimator.
  This example is a great combination of both low level configuration and fast iteration.
  
* The sample provided in [Estimator](./estimator) uses the high level
  `tf.contrib.learn.Estimator` API. This API is great for fast iteration, and
  quickly adapting models to your own datasets without major code overhauls.

* The sample provided in [Keras](./keras) uses the native Keras library.
  This API is great for fast iteration, and quickly adapting models to your own datasets 
  without major code overhauls.
  
* The sample provided in [TFT Transform Estimator](./tfttransformestimator) shows how to use [tf transform](https://github.com/tensorflow/transform) together with [Cloud Dataflow](https://cloud.google.com/dataflow) and [Cloud ML Engine](https://cloud.google.com/ml-engine/).

All the models provided in this directory can be run on the Cloud Machine Learning Engine. To follow along, check out the setup instructions [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

## Notebooks

You can also try the samples in notebooks:

* [Estimator](estimator/trainer/task.ipynb) (Open in [Colab](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/master/census/estimator/trainer/task.ipynb))

* [Custom Estimator](customestimator/trainer/task.ipynb) (Open in [Colab](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/master/census/customestimator/trainer/task.ipynb))

* [Keras](keras/trainer/task.ipynb) (Open in [Colab](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/master/census/keras/trainer/task.ipynb))

* [TensorFlow Core](tensorflowcore/trainer/task.ipynb) (Open in [Colab](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/master/census/tensorflowcore/trainer/task.ipynb))

## Dataset
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

### \*Optional\* Use local training files.

Since TensorFlow - not the Cloud ML Engine - handles reading from GCS, you can run all commands below using these environment variables. However, if your network is slow or unreliable, you may want to download the files for local training.

```
mkdir census_data

gsutil cp $TRAIN_FILE census_data/adult.data.csv
gsutil cp $EVAL_FILE census_data/adult.test.csv

TRAIN_FILE=census_data/adult.data.csv
EVAL_FILE=census_data/adult.test.csv
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
Its worth calling out that unless you want to reuse the old model output dir,
model location should be a new location so that old model doesn't conflict with new
one. 

```
python -m trainer.task --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --job-dir $OUTPUT_DIR \
                       --train-steps $TRAIN_STEPS \
                       --eval-steps 100
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
gcloud ml-engine local train --package-path trainer \
                           --module-name trainer.task \
                           -- \
                           --train-files $TRAIN_FILE \
                           --eval-files $EVAL_FILE \
                           --job-dir $OUTPUT_DIR \
                           --train-steps $TRAIN_STEPS \
                           --eval-steps 100
```

### Using Cloud ML Engine
*NOTE* If you used downloaded the training files to your local file system, be sure
to reset the `TRAIN_FILE` and `EVAL_FILE` environment variables to refer to a GCS location.
Data must be in GCS for cloud-based training.

Run the code on Cloud ML Engine using `gcloud`. Note how `--job-dir` comes
before `--` while training on the cloud and this is so that we can have
different trial runs during Hyperparameter tuning.

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
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-files $TRAIN_FILE \
                                    --eval-files $EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps 100

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
export OUTPUT_DIR=census_$DATE
rm -rf $OUTPUT_DIR
```

```
gcloud ml-engine local train --package-path trainer \
                           --module-name trainer.task \
                           --distributed \
                           -- \
                           --train-files $TRAIN_FILE \
                           --eval-files $EVAL_FILE \
                           --train-steps $TRAIN_STEPS \
                           --job-dir $OUTPUT_DIR \
                           --eval-steps 100

```

### Using Cloud ML Engine
Run the distributed training code on cloud using `gcloud`.

```
export SCALE_TIER=STANDARD_1
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=census_$DATE
export GCS_JOB_DIR=gs://<my-bucket>/path/to/my/models/$JOB_NAME
echo $GCS_JOB_DIR
export TRAIN_STEPS=5000
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --scale-tier $SCALE_TIER \
                                    --runtime-version 1.4 \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-files $TRAIN_FILE \
                                    --eval-files $EVAL_FILE \
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
export HPTUNING_CONFIG=hptuning_config.yaml
export JOB_NAME=census
export TRAIN_STEPS=1000
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --scale-tier $SCALE_TIER \
                                    --runtime-version 1.4 \
                                    --config $HPTUNING_CONFIG \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-files $TRAIN_FILE \
                                    --eval-files $EVAL_FILE \
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
gcloud ml-engine models create census --regions us-central1
```

Then we'll look up the exact path that your exported trained model binaries live in:

```
gsutil ls -r $GCS_JOB_DIR/export
```


 * Estimator Based: You should see a directory named `$GCS_JOB_DIR/export/census/<timestamp>`.
 ```
 export MODEL_BINARIES=$GCS_JOB_DIR/export/census/<timestamp>
 ```

 * Low Level Based: You should see a directory named `$GCS_JOB_DIR/export/JSON/`
   for `JSON`. See other formats `CSV` and `TFRECORD`.
 ```
 export MODEL_BINARIES=$GCS_JOB_DIR/export/JSON/
 ```

```
gcloud ml-engine versions create v1 --model census --origin $MODEL_BINARIES --runtime-version 1.4
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
gcloud ml-engine predict --model census --version v1 --json-instances ../test.json
```

You should see a response with the predicted labels of the examples!

### Run Batch Prediction

If you have large amounts of data, and no latency requirements on receiving prediction results, you can submit a prediction job to the API. This uses the same format as online prediction, but requires data be stored in Google Cloud Storage

```
export JOB_NAME=census_prediction
```

```
gcloud ml-engine jobs submit prediction $JOB_NAME \
    --model census \
    --version v1 \
    --data-format TEXT \
    --region us-central1 \
    --runtime-version 1.4 \
    --input-paths gs://cloud-samples-data/ml-engine/testdata/prediction/census.json \
    --output-path $GCS_JOB_DIR/predictions
```

Check the status of your prediction job:

```
gcloud ml-engine jobs describe $JOB_NAME
```

Once the job is `SUCCEEDED` you can check the results in `--output-path`.
