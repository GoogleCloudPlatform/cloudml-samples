# Molecules
For more details, see [Machine Learning with Apache Beam and TensorFlow](https://cloud.google.com/dataflow/examples/molecules-walkthrough) in the docs.

This sample shows how to create, train, evaluate, and make predictions on a machine learning model, using [Apache Beam](https://beam.apache.org/), [Google Cloud Dataflow](https://cloud.google.com/dataflow/), [TensorFlow](https://www.tensorflow.org/), and [AI Platform](https://cloud.google.com/ai-platform/).

The dataset for this sample is extracted from the [National Center for Biotechnology Information](https://www.ncbi.nlm.nih.gov/) ([FTP source](ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF)).
The file format is [`SDF`](https://en.wikipedia.org/wiki/Chemical_table_file#SDF).
Here's a more detailed description of the [MDL/SDF file format](http://c4.cabrillo.edu/404/ctfile.pdf).

These are the general steps:
 1. Data extraction
 2. Preprocessing the data
 3. Training the model
 4. Doing predictions

## Initial setup

### Getting the source code
You can clone the github repository and then navigate to the `molecules` sample directory.
The rest of the instructions assume that you are in that directory.
```bash
git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git
cd cloudml-samples/molecules
```

### Python virtual environment

> NOTE: It is recommended to run on `python2`.
> Support for using the Apache Beam SDK for Python 3 is in a prerelease state [alpha](https://cloud.google.com/products/?hl=EN#product-launch-stages) and might change or have limited support.

Install a [Python virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments).

Run the following to set up and activate a new virtual environment:
```bash
python2.7 -m virtualenv env
source env/bin/activate
```

Once you are done with the tutorial, you can deactivate the virtual environment by running `deactivate`.

### Installing requirements
You can use the `requirements.txt` to install the dependencies.
```bash
pip install -U -r requirements.txt
```

### Cloud SDK Setup (Optional)

If you have never created application default credentials (ADC), you can create it by `gcloud` command. Following steps are required to `./run-cloud`. 

```bash
gcloud auth application-default login
```

## Quickstart
We'll start by running the end-to-end script locally. To run simply run the [`run-local`](run-local) comand:
```bash
./run-local
```

The script requires a working directory, which is where all the temporary files and other intermediate data will be stored throughout the full run. By default it will use `/tmp/cloudml-samples/molecules` as the working directory. To specify a different working directory, you can specify it via the `--work-dir` option.
```bash
# To use a different local path
./run-local --work-dir ~/cloudml-samples/molecules

# To use a Google Cloud Storage path
./run-local --work-dir gs://<your bucket name here>/cloudml-samples/molecules
```

Each SDF file contains data for 25,000 molecules.
The script will download only 5 SDF files to the working directory by default.
To use a different number of data files, you can use the `--max-data-files` option.
```bash
# To use 10 data files
./run-local --max-data-files 10
```

To run on Google Cloud Platform, all the files must reside in Google Cloud Storage, so you will have to specify a Google Cloud Storage path as the working directory.
To run use the [`run-cloud`](run-cloud) command.
> NOTE: this will incur charges on your Google Cloud Platform project.
```bash
# This will use only 5 data files by default
./run-cloud --work-dir gs://<your-gcs-bucket>/cloudml-samples/molecules
```

## Manual run

### Data Extraction
> Source code: [`data-extractor.py`](data-extractor.py)

This is a data extraction tool to download SDF files from the specified FTP source.
The data files will be stored within a `data` subdirectory inside the working directory.

To store data files locally:
```bash
WORK_DIR=/tmp/cloudml-samples/molecules
python data-extractor.py --work-dir $WORK_DIR --max-data-files 5
```

To store data files to a Google Cloud Storage location:
> NOTE: this will incur charges on your Google Cloud Platform project. See [Storage pricing](https://cloud.google.com/storage/pricing).
```bash
WORK_DIR=gs://<your-gcs-bucket>/cloudml-samples/molecules
python data-extractor.py --work-dir $WORK_DIR --max-data-files 5
```

### Preprocessing
> Source code: [`preprocess.py`](preprocess.py)

This is an [Apache Beam](https://beam.apache.org/) pipeline that will do all the preprocessing necessary to train a Machine Learning model.
It uses [tf.Transform](https://github.com/tensorflow/transform), which is part of [TensorFlow Extended](https://www.tensorflow.org/tfx/), to do any processing that requires a full pass over the dataset.

For this sample, we're doing a very simple feature extraction.
It uses Apache Beam to parse the SDF files and count how many Carbon, Hydrogen, Oxygen, and Nitrogen atoms a molecule has.
To create more complex models we would need to extract more sophisticated features, such as [Coulomb Matrices](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301).

We will eventually train a [Neural Network](https://skymind.ai/wiki/neural-network) to do the predictions.
Neural Networks are more stable when dealing with small values, so it's always a good idea to [normalize](https://en.wikipedia.org/wiki/Feature_scaling) the inputs to a small range (typically from 0 to 1).
Since there's no maximum number of atoms a molecule can have, we have to go through the entire dataset to find the minimum and maximum counts.
Fortunately, tf.Transform integrates with our Apache Beam pipeline and does that for us.

After preprocessing our dataset, we also want to split it into a training and evaluation dataset.
The training dataset will be used to train the model.
The evaluation dataset contains elements that the training has never seen, and since we also know the "answers" (the molecular energy), we'll use these to validate that the training accuracy roughly matches the accuracy on unseen elements.

These are the general steps:
1) Parse the SDF files
2) Feature extraction (count atoms)
3) *Normalization (normalize counts to 0 to 1)
4) Split into 80% training data and 20% evaluation data

> (*) During the normalization step, the Beam pipeline doesn't actually apply the tf.Transform function to our data.
It analyzes the whole dataset to find the values it needs (in this case the minimums and maximums), and with that it creates a TensorFlow graph of operations with those values as constants.
This graph of operations will be applied by TensorFlow itself, allowing us to pass the unnormalized data as inputs rather than having to normalize them ourselves during prediction.

The `preprocess.py` script will preprocess all the data files it finds under `$WORK_DIR/data/`, which is the path where `data-extractor.py` stores the files.

If your training data is small enough, it will be faster to run locally.
```bash
WORK_DIR=/tmp/cloudml-samples/molecules
python preprocess.py --work-dir $WORK_DIR
```

As you want to preprocess a larger amount of data files, it will scale better using [Cloud Dataflow](https://cloud.google.com/dataflow/).
> NOTE: this will incur charges on your Google Cloud Platform project. See [Dataflow pricing](https://cloud.google.com/dataflow/pricing).
```bash
PROJECT=$(gcloud config get-value project)
WORK_DIR=gs://<your-gcs-bucket>/cloudml-samples/molecules
python preprocess.py \
  --project $PROJECT \
  --runner DataflowRunner \
  --temp_location $WORK_DIR/beam-temp \
  --setup_file ./setup.py \
  --work-dir $WORK_DIR
```

### Training the Model
> Source code: [`trainer/task.py`](trainer/task.py)

We'll train a [Deep Neural Network Regressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) in [TensorFlow](https://www.tensorflow.org/).
This will use the preprocessed data stored within the working directory.
During the preprocessing stage, the Apache Beam pipeline transformed extracted all the features (counts of elements) and tf.Transform generated a graph of operations to normalize those features.

The TensorFlow model actually takes the unnormalized inputs (counts of elements), applies the tf.Transform's graph of operations to normalize the data, and then feeds that into our DNN regressor.

For small datasets it will be faster to run locally.
```bash
WORK_DIR=/tmp/cloudml-samples/molecules
python trainer/task.py --work-dir $WORK_DIR

# To get the path of the trained model
EXPORT_DIR=$WORK_DIR/model/export/final
MODEL_DIR=$(ls -d -1 $EXPORT_DIR/* | sort -r | head -n 1)
```

If the training dataset is too large, it will scale better to train on [AI Platform](https://cloud.google.com/ai-platform/).
> NOTE: this will incur charges on your Google Cloud Platform project. See [AI Platform pricing](https://cloud.google.com/ml-engine/docs/pricing).
```bash
JOB="cloudml_samples_molecules_$(date +%Y%m%d_%H%M%S)"
BUCKET=gs://<your-gcs-bucket>
WORK_DIR=gs://<your-gcs-bucket>/cloudml-samples/molecules
gcloud ai-platform jobs submit training $JOB \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket $BUCKET \
  --runtime-version 1.13 \
  --region us-central1 \
  --stream-logs \
  -- \
  --work-dir $WORK_DIR

# To get the path of the trained model
EXPORT_DIR=$WORK_DIR/model/export/final
MODEL_DIR=$(gsutil ls -d "$EXPORT_DIR/*" | sort -r | head -n 1)
```

To visualize the training job, we can use [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
```bash
tensorboard --logdir $WORK_DIR/model
```

You can access the results at [`localhost:6006`](localhost:6006).

### Predictions

#### Option 1: Batch Predictions
> Source code: [`predict.py`](predict.py)

Batch predictions are optimized for throughput rather than latency.
These work best if there's a large amount of predictions to make and you can wait for all of them to finish before having the results.

For batches with a small number of data files, it will be faster to run locally.
```bash
# For simplicity, we'll use the same files we used for training
WORK_DIR=/tmp/cloudml-samples/molecules
python predict.py \
  --work-dir $WORK_DIR \
  --model-dir $MODEL_DIR \
  batch \
  --inputs-dir $WORK_DIR/data \
  --outputs-dir $WORK_DIR/predictions

# To check the outputs
head -n 10 $WORK_DIR/predictions/*
```

For batches with a large number of data files, it will scale better using [Cloud Dataflow](https://cloud.google.com/dataflow/).
```bash
# For simplicity, we'll use the same files we used for training
PROJECT=$(gcloud config get-value project)
WORK_DIR=gs://<your-gcs-bucket>/cloudml-samples/molecules
python predict.py \
  --work-dir $WORK_DIR \
  --model-dir $MODEL_DIR \
  batch \
  --project $PROJECT \
  --runner DataflowRunner \
  --temp_location $WORK_DIR/beam-temp \
  --setup_file ./setup.py \
  --inputs-dir $WORK_DIR/data \
  --outputs-dir $WORK_DIR/predictions

# To check the outputs
gsutil cat "$WORK_DIR/predictions/*" | head -n 10
```

#### Option 2: Streaming Predictions
> Source code: [`predict.py`](predict.py)

Streaming predictions are optimized for latency rather than throughput.
These work best if you are sending sporadic predictions, but want to get the results as soon as possible.

This streaming service will receive molecules from a PubSub topic and publish the prediction results to another PubSub topic.
We'll have to create the topics first.
```bash
# To create the inputs topic
gcloud pubsub topics create molecules-inputs

# To create the outputs topic
gcloud pubsub topics create molecules-predictions
```

For testing purposes, we can start the online streaming prediction service locally.
```bash
# Run on terminal 1
PROJECT=$(gcloud config get-value project)
WORK_DIR=/tmp/cloudml-samples/molecules
python predict.py \
  --work-dir $WORK_DIR \
  --model-dir $MODEL_DIR \
  stream \
  --project $PROJECT \
  --inputs-topic molecules-inputs \
  --outputs-topic molecules-predictions
```

For a highly available and scalable service, it will scale better using [Cloud Dataflow](https://cloud.google.com/dataflow/).
```bash
PROJECT=$(gcloud config get-value project)
WORK_DIR=gs://<your-gcs-bucket>/cloudml-samples/molecules
python predict.py \
  --work-dir $WORK_DIR \
  --model-dir $MODEL_DIR \
  stream \
  --project $PROJECT
  --runner DataflowRunner \
  --temp_location $WORK_DIR/beam-temp \
  --setup_file ./setup.py \
  --inputs-topic molecules-inputs \
  --outputs-topic molecules-predictions
```

Now that we have the prediction service running, we want to run a publisher to send molecules to the streaming prediction service, and we also want a subscriber to be listening for the prediction results.

For convenience, we provided a sample [`publisher.py`](publisher.py) and [`subscriber.py`](subscriber.py) to show how to implement one.

These will have to be run as different processes concurrently, so you'll need to have a different terminal running each command.

> NOTE: remember to activate the `virtualenv` on each terminal.

We'll first run the subscriber, which will listen for prediction results and log them.
```bash
# Run on terminal 2
python subscriber.py \
  --project $PROJECT \
  --topic molecules-predictions
```

We'll then run the publisher, which will parse SDF files from a directory and publish them to the inputs topic.
For convenience, we'll use the same SDF files we used for training.
```bash
# Run on terminal 3
python publisher.py \
  --project $PROJECT \
  --topic molecules-inputs \
  --inputs-dir $WORK_DIR/data
```

Once the publisher starts parsing and publishing molecules, we'll start seeing predictions from the subscriber.

#### Option 3: AI Platform Predictions

If you have a different way to extract the features (in this case the atom counts) that is not through our existing preprocessing pipeline for SDF files, it might be easier to build a JSON file with one request per line and make the predictions on AI Platform.

We've included the [`sample-requests.json`](sample-requests.json) file with an example of how these requests look like. Here are the contents of the file:
```json
{"TotalC": 9, "TotalH": 17, "TotalO": 4, "TotalN": 1}
{"TotalC": 9, "TotalH": 18, "TotalO": 4, "TotalN": 1}
{"TotalC": 7, "TotalH": 8, "TotalO": 4, "TotalN": 0}
{"TotalC": 3, "TotalH": 9, "TotalO": 1, "TotalN": 1}
```

Before creating the model in AI Platform, it is a good idea to test our model's predictions locally:
```bash
# First we have to get the exported model's directory
EXPORT_DIR=$WORK_DIR/model/export/final
if [[ $EXPORT_DIR == gs://* ]]; then
  # If it's a GCS path, use gsutil
  MODEL_DIR=$(gsutil ls -d "$EXPORT_DIR/*" | sort -r | head -n 1)
else
  # If it's a local path, use ls
  MODEL_DIR=$(ls -d -1 $EXPORT_DIR/* | sort -r | head -n 1)
fi

# To do the local predictions
gcloud ai-platform local predict \
  --model-dir $MODEL_DIR \
  --json-instances sample-requests.json
```

For reference, these are the *real* energy values for the `sample-requests.json` file:
```
PREDICTIONS
[37.801]
[44.1107]
[19.4085]
[-0.1086]
```

Once we are happy with our results, we can now upload our model into AI Platform for online predictions.
```bash
# We want the model to reside on GCS and get its path
EXPORT_DIR=$WORK_DIR/model/export/final
if [[ $EXPORT_DIR == gs://* ]]; then
  # If it's a GCS path, use gsutil
  MODEL_DIR=$(gsutil ls -d $EXPORT_DIR/* | sort -r | head -n 1)
else
  # If it's a local path, first upload it to GCS
  LOCAL_MODEL_DIR=$(ls -d -1 $EXPORT_DIR/* | sort -r | head -n 1)
  MODEL_DIR=$BUCKET/cloudml-samples/molecules/model
  gsutil -m cp -r $LOCAL_MODEL_DIR $MODEL_DIR
fi

# Now create the model and a version in AI Platform and set it as default
MODEL=molecules
REGION=us-central1
gcloud ai-platform models create $MODEL --regions $REGION

VERSION="${MODEL}_$(date +%Y%m%d_%H%M%S)"
gcloud ai-platform versions create $VERSION \
  --model $MODEL \
  --origin $MODEL_DIR \
  --runtime-version 1.13

gcloud ai-platform versions set-default $VERSION --model $MODEL

# Finally, we can request predictions via `gcloud ai-platform`
gcloud ai-platform predict \
  --model $MODEL \
  --version $VERSION \
  --json-instances sample-requests.json
```

## Cleanup

Finally, let's clean up all the Google Cloud resources used, if any.

```sh
# To delete the model and version
gcloud ai-platform versions delete $VERSION --model $MODEL
gcloud ai-platform models delete $MODEL

# To delete the inputs topic
gcloud pubsub topics delete molecules-inputs

# To delete the outputs topic
gcloud pubsub topics delete molecules-predictions

# To delete the working directory
gsutil -m rm -rf $WORK_DIR
```
