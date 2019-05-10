# Overview
This code implements a classification model using the Google Cloud Platform. It includes code to process data, train, predict a TensorFlow model 
and assess model performance. This guide trains a neural network model to classify a set of irises in three different classes.

#
* **Data description**

We'll use the
The [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) that this sample
uses for training is hosted by the [UC Irvine Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/) This dataset consists of 150 samples.

The following four numerical features describe their geometrical shape: Sepal Length, Sepal Width, Petal Length and Petal Width.
 
* **Disclaimer**

This dataset is provided by a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.

* **Setup and test your GCP environment**

The best way to setup your GCP project is to use this section in this
[tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#set-up-your-gcp-project).

* **Environment setup:**

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
locally without changing global python packages on your system.

There are two options for the virtual environments:

*   Install [Virtualenv](https://virtualenv.pypa.io/en/stable/) 
    *   Create virtual environment `virtualenv myvirtualenv`
    *   Activate env `source myvirtualenv/bin/activate`
*   Install [Miniconda](https://conda.io/miniconda.html)
    *   Create conda environment `conda create --name myvirtualenv python=2.7`
    *   Activate env `source activate myvirtualenv`

* **Install dependencies**

Install the python dependencies. `pip install --upgrade -r requirements.txt`

#

* **How to satisfy AI Platform project structure requirements**

Follow [this](https://cloud.google.com/ml-engine/docs/tensorflow/packaging-trainer#project-structure) guide to structure your training application.


<h1>Data processing</h1>

We host the Iris files required for you to run this sample. Files are hosted in a Public Bucket.

If you want to use local files directly, you can use the following commands:

```
IRIS_DATA=data
mkdir $IRIS_DATA
gsutil cp gs://cloud-samples-data/ml-engine/iris/iris_training.csv $IRIS_DATA
gsutil cp gs://cloud-samples-data/ml-engine/iris/iris_test.csv $IRIS_DATA
```

* **Upload the data to a Google Cloud Storage bucket**

AI Platform works by using resources available in the cloud, so the training
data needs to be placed in such a resource. For this example, we'll use [Google
Cloud Storage], but it's possible to use other resources like [BigQuery]. Make a
bucket (names must be globally unique) and place the data in there:

```shell
gsutil mb gs://your-bucket-name  # Change your BUCKET
gsutil cp -r data/iris_training.csv gs://your-bucket-name/iris_training.csv
gsutil cp -r data/iris_test.csv gs://your-bucket-name/iris_test.csv
```

<h1>Training</h1>

* **GCloud configuration:**

```
mkdir data
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=iris_$DATE
export TRAIN_FILE=$DATA/iris_training.csv
export EVAL_FILE=$DATA/iris_test.csv
rm -rf $JOB_DIR
```

* **Test locally:**

```
python -m trainer.task \
 --train-file=$TRAIN_FILE \
 --job-dir=$JOB_DIR
```

* **AI Platform**

* **GCloud local configuration:**

```
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=iris_$DATE
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/iris/iris_training.csv
export EVAL_FILE=gs://cloud-samples-data/ml-engine/iris/iris_test.csv
export TRAIN_STEPS=1000
export EVAL_STEPS=100
rm -rf $JOB_DIR
```

* **Run locally via the gcloud command for AI Platform:**

```
gcloud ml-engine local train --package-path trainer \
    --module-name trainer.task \
    -- \
    --train-file $TRAIN_FILE \
    --eval-file $EVAL_FILE \
    --job-dir $JOB_DIR \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS
```

* **Run in AI Platform**

You can train the model on AI Platform:

*NOTE:* If you downloaded the training files to your local filesystem, be sure
to reset the `TRAIN_FILE` and `EVAL_FILE` environment variables to refer to a GCS location.
Data must be in GCS for cloud-based training.

Run the code on AI Platform using `gcloud`. Note how `--job-dir` comes
before `--` while training on the cloud and this is so that we can have
different trial runs during Hyperparameter tuning.

* **GCloud configuration:**

```
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=iris_$DATE
export GCS_JOB_DIR=gs://your-bucket-name/path/to/my/jobs/$JOB_NAME  # Change your BUCKET
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/iris/iris_training.csv
export EVAL_FILE=gs://cloud-samples-data/ml-engine/iris/iris_test.csv
export TRAIN_STEPS=1000
export EVAL_STEPS=100
export REGION=us-central1
export SCALE_TIER=STANDARD_1
```

* **Run in AI Platform:**

```
gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --runtime-version 1.13 \
    --job-dir $GCS_JOB_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --train-file $TRAIN_FILE \
    --eval-file $EVAL_FILE \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS
```

* **Distributed Node Training in AI Platform:**

Distributed node training uses [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).
The main change to make the distributed version work is usage of [TF_CONFIG](https://cloud.google.com/ml/reference/configuration-data-structures#tf_config_environment_variable)
environment variable. The environment variable is generated using `gcloud` and parsed to create a
[ClusterSpec](https://www.tensorflow.org/deploy/distributed#create_a_tftrainclusterspec_to_describe_the_cluster). See the [ScaleTier](https://cloud.google.com/ml/pricing#ml_training_units_by_scale_tier) for predefined tiers

* **Run locally:**

```
gcloud ml-engine local train --package-path trainer \
    --module-name trainer.task \
    --distributed \
    -- \
    --train-file $TRAIN_FILE \
    --eval-file $EVAL_FILE \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --job-dir $JOB_DIR

```
                        
* **Run in AI Platform:**

```
gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --scale-tier $SCALE_TIER \
    --runtime-version 1.13 \
    --job-dir $GCS_JOB_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --train-file $TRAIN_FILE \
    --eval-file $EVAL_FILE \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS
```

# Hyperparameter Tuning

```
export HPTUNING_CONFIG=hptuning_config.yaml
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --scale-tier $SCALE_TIER \
    --runtime-version 1.13 \
    --config $HPTUNING_CONFIG \
    --job-dir $GCS_JOB_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --train-file $TRAIN_FILE \
    --eval-file $EVAL_FILE \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS                        
```

# Prediction

Once your training job has finished, you can use the exported model to create a prediction server. To do this you first create a model:

```
gcloud ml-engine models create iris --regions us-central1
```

Then we'll look up the exact path that your exported trained model binaries live in:

```
gsutil ls -r $GCS_JOB_DIR/export
```


 * Estimator Based: You should see a directory named: `$GCS_JOB_DIR/export/iris/<timestamp>`.
```
export MODEL_BINARIES=$GCS_JOB_DIR/export/iris/<timestamp>
```

 * Low Level Based: You should see a directory named `$JOB_DIR/export/JSON/`
   for `JSON`. See other formats `CSV` and `TFRECORD`.
 
```
export MODEL_BINARIES=$GCS_JOB_DIR/export/CSV/
```

```
gcloud ml-engine versions create v1 \
    --model iris \
    --origin $MODEL_BINARIES \
    --runtime-version 1.13
```

(Optional) Inspect the model binaries with the SavedModel CLI
TensorFlow ships with a CLI that allows you to inspect the signature of exported binary files. To do this run:

```
SIGNATURE_DEF_KEY=`saved_model_cli show --dir $MODEL_BINARIES --tag serve | grep "SignatureDef key:" | awk 'BEGIN{FS="\""}{print $2}' | head -1`
saved_model_cli show --dir $MODEL_BINARIES --tag serve --signature_def $SIGNATURE_DEF_KEY
```

* **Run Online Predictions**

You can now send prediction requests to the API. To test this out you can use the `gcloud ml-engine predict` tool:

```
gcloud ml-engine predict --model iris --version v1 --json-instances test.json
```

Using CSV:

```
gcloud ml-engine predict --model iris --version v1 --text-instances test.csv
```

You should see a response with the predicted labels of the examples!

* **Run Batch Prediction**

```
export JOB_NAME=iris_prediction
```

```
gcloud ml-engine jobs submit prediction $JOB_NAME \
    --model iris \
    --version v1 \
    --data-format TEXT \
    --region $REGION \
    --runtime-version 1.13 \
    --input-paths gs://cloud-samples-data/ml-engine/testdata/prediction/iris.json \
    --output-path $GCS_JOB_DIR/predictions
```

Check the status of your prediction job:

```
gcloud ml-engine jobs describe $JOB_NAME
```

Once the job is `SUCCEEDED` you can check the results in `--output-path`.


* **Monitor with TensorBoard:**

```
tensorboard --logdir=$GCS_JOB_DIR
```

## References

[Tensorflow tutorial](https://www.tensorflow.org/guide/premade_estimators)
