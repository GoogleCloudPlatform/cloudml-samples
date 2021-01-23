# Overview
This code implements a Logistic Regression model using the Google Cloud Platform. 
It includes code to process data, train a tensorflow model with hyperparameter tuning, run predictions on new data and assess model performance.

#
*  **Examples**
 
There are six samples provided in this directory. They allow you to move from
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
  `tf.estimator.DNNLinearCombinedClassifier` API. This API is great for fast iteration, and
  quickly adapting models to your own datasets without major code overhauls.

* The sample provided in [Keras](./keras) uses the native Keras library.
  This API is great for fast iteration, and quickly adapting models to your own datasets 
  without major code overhauls.

* The sample provided in [TensorFlow Keras](./tf-keras) uses `tf.keras`,
  TensorFlow's implementation of the Keras API specification. This provides the
  benefits of Keras and also first-class support for TensorFlow-specific
  functionality.

* The sample provided in [TFT Transform Estimator](./tftransformestimator) shows how to use [tf transform](https://github.com/tensorflow/transform) together with [Cloud Dataflow](https://cloud.google.com/dataflow) and [AI Platform](https://cloud.google.com/ml-engine/).

All the models provided in this directory can be run on AI Platform.

#
* **Notebooks**

    - [TensorFlow Keras](../notebooks/tensorflow/getting-started-keras.ipynb) (Open in [Colab](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/master/notebooks/tensorflow/getting-started-keras.ipynb))

#
* **Data description**

The [Census Income Data
Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this sample
uses for training is hosted by the [UC Irvine Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/).

Using census data which contains data a person's age, education, marital status, 
and occupation (the features), we will try to predict whether or not the person earns 
more than 50,000 dollars a year (the target label). We will train a logistic regression 
model that, given an individual's information, outputs a number between 0 and 1, this can 
be interpreted as the probability that the individual has an annual income of over 50,000 dollars.

As a modeler and developer, think about how this data is used and the potential benefits and harm a model's predictions can cause. A model like this could reinforce societal biases and disparities. Is each feature relevant to the problem you want to solve or will it introduce bias? For more information, read about [ML fairness](https://developers.google.com/machine-learning/fairness-overview/).
 
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



# Data processing

We host the Census files required for you to run this sample. Files are hosted in a Public Bucket.

If you want to use local files directly, you can use the following commands:

TensorFlow - not AI Platform - handles reading from GCS, you can run all commands below using these environment variables. However, if your network is slow or unreliable, you may want to download the files for local training.

```
CENSUS_DATA=census_data
mkdir $CENSUS_DATA
gsutil cp gs://cloud-samples-data/ml-engine/census/data/adult.data.csv $CENSUS_DATA
gsutil cp gs://cloud-samples-data/ml-engine/census/data/adult.test.csv $CENSUS_DATA
```

* **Upload the data to a Google Cloud Storage bucket**

AI Platform works by using resources available in the cloud, so the training
data needs to be placed in such a resource. For this example, we'll use [Google
Cloud Storage], but it's possible to use other resources like [BigQuery]. Make a
bucket (names must be globally unique) and place the data in there:

```shell
gsutil mb gs://your-bucket-name
gsutil cp -r data/adult.data.csv gs://your-bucket-name/adult.data.csv
gsutil cp -r data/adult.test.csv gs://your-bucket-name/adult.test.csv
```

# Training

* **GCloud configuration:**

```
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=census_$DATE
export TRAIN_FILE=census_data/adult.data.csv
export EVAL_FILE=census_data/adult.test.csv
export TRAIN_STEPS=1000
rm -rf $JOB_DIR
```

* **Test locally:**

```
python -m trainer.task --train-files $TRAIN_FILE \
    --eval-files $EVAL_FILE \
    --job-dir $JOB_DIR \
    --train-steps $TRAIN_STEPS \
    --eval-steps 100
```

* **AI Platform**

* **GCloud local configuration:**

```
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=census_$DATE
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv
export EVAL_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.test.csv
export TRAIN_STEPS=1000
rm -rf $JOB_DIR
```

* **Run locally via the gcloud command for AI Platform:**

```
gcloud ai-platform local train --package-path trainer \
    --module-name trainer.task \
    -- \
    --train-files $TRAIN_FILE \
    --eval-files $EVAL_FILE \
    --job-dir $JOB_DIR \
    --train-steps $TRAIN_STEPS \
    --eval-steps 100
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
export JOB_NAME=census_$DATE
export BUCKET_NAME=your-bucket-name  # TODO: Change your BUCKET
export GCS_JOB_DIR=gs://$BUCKET_NAME/path/to/my/jobs/$JOB_NAME  
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv
export EVAL_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.test.csv
export TRAIN_STEPS=5000
export REGION=us-central1
```

* **Run in AI Platform:**

```
gcloud ai-platform jobs submit training $JOB_NAME \
    --stream-logs \
    --runtime-version 1.15 \
    --job-dir $GCS_JOB_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --train-files $TRAIN_FILE \
    --eval-files $EVAL_FILE \
    --train-steps $TRAIN_STEPS \
    --eval-steps 100
```

* **Monitor with TensorBoard:**

```
tensorboard --logdir=$GCS_JOB_DIR
```

* **Accuracy and Output:**

You should see the output for default number of training steps and approx accuracy close to `80%`.

* **Distributed Node Training in AI Platform:**

Distributed node training uses [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).
The main change to make the distributed version work is usage of [TF_CONFIG](https://cloud.google.com/ml/reference/configuration-data-structures#tf_config_environment_variable)
environment variable. The environment variable is generated using `gcloud` and parsed to create a
[ClusterSpec](https://www.tensorflow.org/deploy/distributed#create_a_tftrainclusterspec_to_describe_the_cluster). See the [ScaleTier](https://cloud.google.com/ml/pricing#ml_training_units_by_scale_tier) for predefined tiers.

* **GCloud configuration:**

```
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=census_$DATE
export BUCKET_NAME=your-bucket-name  # TODO: Change your BUCKET
export GCS_JOB_DIR=gs://$BUCKET_NAME/path/to/my/jobs/$JOB_NAME
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv
export EVAL_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.test.csv
export TRAIN_STEPS=5000
export REGION=us-central1
export SCALE_TIER=STANDARD_1
```

* **Run locally:**

```
gcloud ai-platform local train --package-path trainer \
    --module-name trainer.task \
    --distributed \
    -- \
    --train-files $TRAIN_FILE \
    --eval-files $EVAL_FILE \
    --train-steps $TRAIN_STEPS \
    --job-dir $GCS_JOB_DIR \
    --eval-steps 100
```
                        
* **Run in AI Platform:**

```
gcloud ai-platform jobs submit training $JOB_NAME \
    --stream-logs \
    --scale-tier $SCALE_TIER \
    --runtime-version 1.15 \
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

AI Platform allows you to perform Hyperparameter tuning to find out the
most optimal hyperparameters. See [Overview of Hyperparameter Tuning]
(https://cloud.google.com/ml/docs/concepts/hyperparameter-tuning-overview) for more details.

```
export HPTUNING_CONFIG=hptuning_config.yaml
```

Running Hyperparameter job is almost exactly same as Training job except that
you need to add the `--config` argument.

```
gcloud ai-platform jobs submit training $JOB_NAME \
    --stream-logs \
    --scale-tier $SCALE_TIER \
    --runtime-version 1.15 \
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
gcloud ai-platform models create census --regions us-central1
```

Then we'll look up the exact path that your exported trained model
binaries live in:

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
export MODEL_BINARIES=$GCS_JOB_DIR/export/CSV/
```

```
gcloud ai-platform versions create v1 --model census --origin $MODEL_BINARIES --runtime-version 1.13

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
gcloud ai-platform predict --model census --version v1 --json-instances test.json
```

Using CSV:

```
gcloud ai-platform predict --model census --version v1 --text-instances test.csv
```

You should see a response with the predicted labels of the examples!

* **Run Batch Prediction**

```
export JOB_NAME=census_prediction
```

```
gcloud ai-platform jobs submit prediction $JOB_NAME \
    --model census \
    --version v1 \
    --data-format TEXT \
    --region $REGION \
    --runtime-version 1.15 \
    --input-paths gs://cloud-samples-data/ml-engine/testdata/prediction/census.json \
    --output-path $GCS_JOB_DIR/predictions
```

Check the status of your prediction job:

```
gcloud ai-platform jobs describe $JOB_NAME
```

Once the job is `SUCCEEDED` you can check the results in `--output-path`.


# (Optional) Preprocessing with Dataflow

**Note: This is available only for [Estimator](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/estimator)/.**

* **Objective**

Data preprocessing is not an absolute necessity in our use case. You can see running instructions without preprocessing: in this case, we are training the model on train.csv, evaluating it on test.csv and keeping one example in test.json to call the deployed model with.

One minor caveat is that we do not have a perfect train/eval/test split: we are using the real test set as a validation one and have only one example in the real test set, which is enough to debug the model API but not enough to run complementary analysis. Hence, we build a data preprocessing pipeline that will correct this by splitting the initial train csv into a train and eval sets. 

We will implement this in Dataflow, which like using a sledgehammer to crack a nut, however this is a great opportunity to showcase the key principles of this tool. In other applications, it is likely that your initial data will require some cleaning with Dataflow.


* **GCloud configuration:**

```
export BUCKET_NAME=your-bucket-name
export PROJECT_ID=$(gcloud config list --format 'value(core.project)' 2>/dev/null)

export TRAINING_DATA=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv
```

* **Run preprocessing locally**

```
DATAFLOW_DIR=dataflow_dir

python -m preprocessing/run_preprocessing \
    --project_id $PROJECT_ID \
    --job_dir $DATAFLOW_DIR \
    --input_data $TRAINING_DATA
```

* **Run preprocessing on Dataflow**

```
DATE_TIME=$(date +"%Y%m%d_%H%M%S")
DATAFLOW_DIR=gs://$BUCKET_NAME/preprocessing/${JOB_NAME}
JOB_NAME=preprocessing-${DATE_TIME}-${USER}

python -m preprocessing/run_preprocessing \
    --project_id $PROJECT_ID \
    --job_name $JOB_NAME \
    --job_dir $DATAFLOW_DIR \
    --input_data $TRAINING_DATA \
    --cloud
```


* **Use the updated train and eval files**

```
export TRAIN_FILE=${DATAFLOW_DIR}/output_data/train*.csv
export EVAL_FILE=${DATAFLOW_DIR}/output_data/eval*.csv
```

## References

[Tensorflow tutorial](https://www.tensorflow.org/guide/premade_estimators)
