# Molecules
This sample shows how to train and create an ML model to predict the molecular energy, including how to preprocess raw data files.

The dataset for this sample comes from this [Kaggle Dataset](https://www.kaggle.com/burakhmmtgl/predict-molecular-properties). However, the number of preprocessed JSON files was too small, so this sample will download the raw data files directly from this [FTP source](ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF) (in [`SDF`](https://en.wikipedia.org/wiki/Chemical_table_file#SDF) fromat instead of `JSON`). Here's a more detailed description of the [MDL/SDF file format](http://c4.cabrillo.edu/404/ctfile.pdf).

These are the rough steps:
 1. data-extractor.py: extracts the data files
 2. preprocess.py: runs an [Apache Beam](https://beam.apache.org/) pipeline for element-wise transformations, and [tf.Transform](https://github.com/tensorflow/transform) for full-pass transformations. This can be run in [Google Cloud Dataflow](https://cloud.google.com/dataflow/)
 4. trainer/task.py: trains and evaluates the ([Tensorflow](https://www.tensorflow.org/)) model. This can be run in [Google Cloud ML Engine](https://cloud.google.com/ml-engine/)

This model only does a very simple preprocessing. It uses Apache beam to parse the SDF files and count how many Carbon, Hydrogen, Oxygen, and Nitrogen atoms a molecule has. Then it uses tf.Transform to normalize to values between 0 and 1. Finally, the normalized counts are fed into a TensorFlow Deep Neural Network. There are much more interesting features that could be extracted that will make more accurate predictions.

## Initial setup
> NOTE: This requires `python2`, Apache Beam does not currently support `python3`.

### Getting the source code
You can clone the github repository and then navigate to the `molecules` sample directory. The rest of the instructions assume that you are in that directory.
```bash
git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git
cd cloudml-samples/molecules
```

### Python virtual environment
Using [virtualenv](https://virtualenv.pypa.io/en/stable/) to isolate your dependencies is recommended. To set up, make
sure you have the `virtualenv` package installed.
```bash
pip install --user virtualenv
```

To create and activate a new virtual environment, run the following commands:
```bash
python -m virtualenv env
source env/bin/activate
```

To deactivate the virtual environment, run:
```bash
deactivate
```

See [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) for details.

### Installing requirements
You can use the `requirements.txt` to install the dependencies.
```bash
pip install -r requirements.txt
```

## Running locally
By default, all the scripts will store temporary data into `/tmp/cloudml-samples/molecules/`. Also, by default, this will only use 5 data files, each of them containing 250,000 molecules.
```bash
# Extract the data files
python data-extractor.py

# Preprocess the datasets
python preprocess.py

# Train and evaluate the model
python trainer/task.py

# Get the model path
MODEL_DIR=$(ls -d -1 /tmp/cloudml-samples/molecules/model/export/molecules/* | sort -r | head -n 1)
echo "Model: $MODEL_DIR"

# Make local predictions
gcloud ml-engine local predict \
  --model-dir $MODEL_DIR \
  --json-instances sample-requests.json
```

An end-to-end script has been included for your convenience, where you can specify a different number of data files using the `--total-data-files` option, as well as a different working directory using the `--work-dir` option.
```bash
# Simple run
bash run-local

# Run in your home directory
bash run-local --work-dir ~/cloudml-samples/molecules
```

For reference, this are the *real* energy values for the `sample-requests.json` file.
```bash
PREDICTIONS
[37.801]
[44.1107]
[19.4085]
[-0.1086]
```

## Running in Google Cloud
To run on Google Cloud, all the files must reside in Google Cloud Storage. We'll start by defining our work directory.
```bash
WORK_DIR=gs://<Your bucket name>/cloudml-samples/molecules
```

After specifying our work directory, we can then extract the data files, preprocess, and train in Google Cloud using that location.
```bash
# Extract the data files
DATA_DIR=$WORK_DIR/data
python data-extractor.py \
  --data-dir $DATA_DIR \
  --total-data-files 10

# Preprocess the datasets using Apache Beam's DataflowRunner
PROJECT=$(gcloud config get-value project)
TEMP_DIR=$WORK_DIR/temp
PREPROCESS_DATA=$WORK_DIR/PreprocessData
python preprocess.py \
  --data-dir $DATA_DIR \
  --temp-dir $TEMP_DIR \
  --preprocess-data $PREPROCESS_DATA \
  --runner DataflowRunner \
  --project $PROJECT \
  --temp_location $TEMP_DIR \
  --setup_file ./setup.py

# Train and evaluate the model in Google ML Engine
JOB="cloudml_samples_molecules_$(date +%Y%m%d_%H%M%S)"
BUCKET=$(echo $WORK_DIR | egrep -o gs://[-_.a-z0-9]+)
EXPORT_DIR=$WORK_DIR/model
gcloud ml-engine jobs submit training $JOB \
  --stream-logs \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket $BUCKET \
  -- \
  --preprocess-data $PREPROCESS_DATA \
  --export-dir $EXPORT_DIR

# Get the model path
MODEL_DIR=$(gsutil ls -d $EXPORT_DIR/export/molecules/* | sort -r | head -n 1)
echo "Model: $MODEL_DIR"

# Create a model in Google Cloud ML Engine
MODEL=molecules
gcloud ml-engine models create $MODEL

# Create a model version
VERSION=$JOB
gcloud ml-engine versions create $VERSION \
  --model $MODEL \
  --origin $MODEL_DIR

# Make predictions
gcloud ml-engine predict \
  --model $MODEL \
  --version $VERSION \
  --json-instances sample-requests.json
```

There's also an end-to-end script for a cloud run. You can also specify the number of data files with the `--total-data-files` option, and the `--work-dir` has to be specified to a Google Cloud Storage location.
```bash
WORK_DIR=gs://<Your bucket name>/cloudml-samples/molecules

# Simple run
bash run-cloud --work-dir $WORK_DIR

# Run using 10 data files
bash run-cloud --work-dir $WORK_DIR --total-data-files 10
```

For reference, this are the *real* energy values for the `sample-requests.json` file.
```bash
PREDICTIONS
[37.801]
[44.1107]
[19.4085]
[-0.1086]
```
