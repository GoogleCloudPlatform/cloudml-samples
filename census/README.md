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

The local training environment for Cloud ML uses your local python installation. So you will need to have Python 2.7 and install TensorFlow following the instructions [here](https://www.tensorflow.org/versions/r1.0/get_started/os_setup). (Note please ensure that you install `tensorflow>=1.0.0-rc2`)

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

## Local Prediction

If you wish, you may validate that the model will accept the recommended inputs when you deploy it to the prediction service. You can do this with the `gcloud beta ml local predict` tool. First you must install the Cloud ML SDK.

```
pip install --upgrade --force-reinstall \
    https://storage.googleapis.com/cloud-ml/sdk/cloudml.latest.tar.gz
```

Then run:

```
gcloud beta ml local predict --model-dir=output/export/Servo/<TIMESTAMP>/ --json-instances=test.json
```

## Run On The Cloud

NOTE: Cloud ML Training Jobs are regional. If you do not want to use the us-central1 region, you will need to copy the training data to a regional bucket that matches the region you wish to choose.

Below we will use `OUTPUT_PATH` to represent the fully qualified GCS location for model checkpoints, summaries, and exports. 

### Single Worker

```
gcloud beta ml jobs submit training census \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.0 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region us-central1
    -- \
    --train-file gs://tf-ml-workshop/widendeep/adult.data \
    --eval-file gs://tf-ml-workshop/widendeep/adult.test \
    --train-steps 1000
```

### Distributed

TF.Learn models require no code changes to be distributed in Cloud ML. Simply add the `--scale-tier STANDARD_1` flag (or any of the other scale tiers above basic). (Again, be sure to add this above the `--` flag).

### HyperParameter Tuning

Coming Soon

## Create a Prediction Server

Once your training job has finished, you can use the exported model to create a prediction server. To do this you first create a model:

```
gcloud beta ml models create census --regions us-central1
```

Then create a version for that model:

```
gcloud beta ml versions create v1 --model census --origin $OUTPUT_PATH/export
```

You can now send prediction requests to the API. To test this out you can use the `gcloud beta ml predict` tool:

```
gcloud beta ml predict --model census --version v1 --json-instances test.json
```

You should see a response with the predicted labels of the examples!
