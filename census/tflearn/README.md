# Census: TF.Learn Distributed Training and Prediction

This sample walks you through training and prediction, including distributed training, and hyperparameter tuning.

## Local Training

### Getting Data

Throughout we will use a census dataset for training. For local training, you can read them directly from Google Cloud Storage or if you have an unreliable network, can download them to your local file system.

```
TRAIN_FILE=gs://tf-ml-workshop/widendeep/adult.data
EVAL_FILE=gs://tf-ml-workshop/widendeep/adult.test
```

To use locally simply copy them down with gsutil:
```
gsutil cp gs://tf-ml-workshop/widendeep/* data/
```

And use the local locations whereever the environment variables above are used.


### Installing Dependencies

The local training environment for Cloud ML uses your local python installation. So you will need to have Python 2.7 and install TensorFlow following the instructions [here](https://www.tensorflow.org/get_started/os_setup).

### Running Training

Then you can run training with the following command:

```
gcloud beta ml local train \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --train-file $TRAIN_FILE \
    --eval-file $EVAL_FILE \
    --train-steps 1000 \
    --job_dir output
```

You should see very verbose output, ending with a line about saving your model.

To ensure that your model works with Cloud MLs distributed execution environment you can add the `--distributed` flag (before the `--` that seperates user args from CLI args). You will see output from a number of different processes which are communicating through gRPC!

## Run On The Cloud

NOTE: Cloud ML Training Jobs are regional. If you do not want to use the us-central1 region, you will need to copy the training data to a regional bucket that matches the region you wish to choose.

Below we will use `OUTPUT_PATH` to represent the fully qualified GCS location for model checkpoints, summaries, and exports. So for example:

```
OUTPUT_PATH=gs://<my-bucket>/path/to/my/models/run3
```

### Single Worker

```
gcloud beta ml jobs submit training census \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.0 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region us-central1 \
    -- \
    --train-file gs://tf-ml-workshop/widendeep/adult.data \
    --eval-file gs://tf-ml-workshop/widendeep/adult.test \
    --train-steps 1000
```

### Distributed

TF.Learn models require no code changes to be distributed in Cloud ML. Simply add the `--scale-tier STANDARD_1` flag (or any of the other scale tiers above basic). (Again, be sure to add this above the `--` flag).

### HyperParameter Tuning

To use hyperparameter tuning, we must define an hyperparameter tuning config as described [here](https://cloud.google.com/ml/docs/how-tos/using-hyperparameter-tuning) and pass it to `gcloud` using the `--config` flag. An example config that tunes embedding sizes for the deep columns of this model is provided in [hptuning_config.yaml](hptuning_config.yaml).

The necessary code changes to enable hyperparameter tuning are minimal, and have already been made. They are:

1. Create a metric using `tf.contrib.learn.MetricSpec` named `training/hptuning/metric` and pass it to `Experiment` to be used in evaluation. Here we chose `tf.contrib.metrics.streaming_accuracy` as our metric.

2. Make sure to use the environment variable `TF_CONFIG['task']['trial']` to scope your model directories so that different replicas do not write and read checkpoints from the same locations.


## Create a Prediction Server

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
gcloud beta ml versions create v1 --model census --origin $MODEL_BINARIES
```

You can now send prediction requests to the API. To test this out you can use the `gcloud beta ml predict` tool:

```
gcloud beta ml predict --model census --version v1 --json-instances test.json
```

You should see a response with the predicted labels of the examples!
