# Census: Estimator-based Distributed Training and Prediction

This sample walks you through training and prediction, including distributed training, and hyperparameter tuning.

All commands below assume you have cloned this repository and changed to the current directory. It also assumes you have set up the Cloud ML Engine as described [here](https://cloud-dot-devsite.googleplex.com/ml/docs/how-tos/getting-set-up).

## Local Training

### Getting Data

The [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this sample uses for training is hosted by the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/)

We have done some trivial cleaning and hosted it at the following locations:

```
TRAIN_FILE=gs://cloudml-public/census/data/adult.data.csv
EVAL_FILE=gs://cloudml-public/census/data/adult.test.csv
```

For local training, you can read them directly from Google Cloud Storage or if you have an unreliable network, can download them to your local file system.

```
mkdir data
gsutil cp gs://tf-ml-workshop/widendeep/* data/
```

And use the local locations wherever the environment variables above are used.

### Installing Dependencies

The local training environment for Cloud ML uses your local python installation. So you will need to have Python 2.7 and install TensorFlow following the instructions [here](https://www.tensorflow.org/get_started/os_setup).

### Running Training

Then you can run training with the following command:

```
gcloud beta ml local train \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --train-files $TRAIN_FILE \
    --eval-files $EVAL_FILE \
    --train-steps 1000 \
    --job-dir output
```

You should see very verbose output, ending with a line about saving your model.

To ensure that your model works with Cloud MLs distributed execution environment you can add the `--distributed` flag (before the `--` that seperates user args from CLI args). You will see output from a number of different processes which are communicating through gRPC!

## Run On The Cloud

NOTE: Cloud ML Training Jobs are regional. If you do not want to use the us-central1 region, you will need to copy the training data to a regional bucket that matches the region you wish to choose.

Below we will use `OUTPUT_PATH` to represent the fully qualified GCS location for model checkpoints, summaries, and exports. So for example:

```
OUTPUT_PATH=gs://<my-bucket>/path/to/my/models/run3
```

Additionally, if you set `TRAIN_FILES` and `EVAL_FILES` to local locations you will need to update them to a GCS location.

### Single Worker

```
JOB_NAME=census
gcloud beta ml jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH/$JOB_NAME \
    --runtime-version 1.0 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region us-central1 \
    -- \
    --train-files $TRAIN_FILES \
    --eval-files $EVAL_FILES \
    --train-steps 1000
```

NOTE: to restart a failed job you may simply use the same `--job-dir` location (and alter `JOB_NAME`), and the parameters will be loaded from the last saved checkpoint.


### Distributed

TF.Learn models require no code changes to be distributed in Cloud ML. Simply add the `--scale-tier STANDARD_1` flag (or any of the other [scale tiers](https://cloud.google.com/ml/reference/rest/v1beta1/projects.jobs#scaletier) above basic). (Again, be sure to add this above the `--` flag).

### HyperParameter Tuning

To use hyperparameter tuning, we must define an hyperparameter tuning config as described [here](https://cloud.google.com/ml/docs/how-tos/using-hyperparameter-tuning) and pass it to `gcloud` using the `--config` flag. An example config that tunes the hidden layers of the `DNN` is provided in [hptuning_config.yaml](hptuning_config.yaml).

To use hyperparameter tuning we need only create a metric using `tf.contrib.learn.MetricSpec` and pass it to `Experiment` to be used in evaluation. Here our `DNNLinearCombinedClassifier` comes with a number of default metrics, we'll use `accuracy`.

Additionally, we must ensure that different runs use different directories for checkpoints summaries and exports. Fortunately this is taken care of by passing `--job-dir` to the gcloud command rather than directly to our code. The service will use this as a base directory in hyperparameter tuning and append trial ids to the path.

## Run Predictions

### Create A Prediction Service

Once your training job has finished, you can use the exported model to create a prediction server. To do this you first create a model:

```
gcloud beta ml models create census --regions us-central1
```

Then we'll look up the exact path that your exported trained model binaries live in:

```
gsutil ls -r $OUTPUT_PATH/export
```

You should see a directory named `$OUTPUT_PATH/export/Servo/<timestamp>`. Copy this directory path (without the `:` at the end) and set the environment variable `MODEL_BINARIES` to it's value. Then run:


```
gcloud beta ml versions create v1 --model census --origin $MODEL_BINARIES --runtime-version 1.0
```

### Run Online Predictions

You can now send prediction requests to the API. To test this out you can use the `gcloud beta ml predict` tool:

```
gcloud beta ml predict --model census --version v1 --json-instances test.json
```

You should see a response with the predicted labels of the examples!

### Run Batch Prediction

If you have large amounts of data, and no latency requirements on receiving prediction results, you can submit a prediction job to the API. This uses the same format as online prediction, but requires data be stored in Google Cloud Storage

```
gcloud beta ml jobs submit prediction my_prediction_job \
    --model census \
    --version v1 \
    --data-format TEXT \
    --region us-central1 \
    --input-paths gs://cloudml-public/testdata/prediction/census.json \
    --output-path $OUTPUT_PATH/predictions
```
