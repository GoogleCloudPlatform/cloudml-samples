# Molecules
This sample shows how to train and create an ML model to predict the energy on molecules, including how to preprocess raw data files.

The dataset for this sample comes from this [Kaggle Dataset](https://www.kaggle.com/burakhmmtgl/predict-molecular-properties). However, the number of preprocessed JSON files was too small, so this sample will download the raw data files directly from this [FTP source](ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF) (in [`SDF`](https://en.wikipedia.org/wiki/Chemical_table_file#SDF) fromat instead of `JSON`). Here's a more detailed description of the [MDL/SDF file format](http://c4.cabrillo.edu/404/ctfile.pdf).

Roughly the steps are the following:
 1. Validate the command line arguments
 2. Extract the raw data
 3. Preprocessing ([Apache Beam](https://beam.apache.org/) for element-wise transformations, and [tf.Transform](https://github.com/tensorflow/transform)) for transformations that require a full pass of the dataset. This can be run distributed in [Google Cloud Dataflow](https://cloud.google.com/dataflow/)
 4. Training ([Tensorflow](https://www.tensorflow.org/)), this can be run distributed in [Google Cloud ML Engine](https://cloud.google.com/ml-engine/)

## Installing
This requires `python2`, Apache Beam does not currently support `python3`.

### Getting the source code
You can clone the github repository and then navigate to the `molecules` sample directory. The rest of the instructions assume that you are in that directory.
```bash
git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git
cd cloudml-samples/molecules
```

### Python virtual environment
Using [virtualenv](https://virtualenv.pypa.io/en/stable/) to isolate your dependencies is recommended. To set up make
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
You can run locally a small job by executing:
```bash
python main.py
```

The data sets and any other temporary file will be written to the working directory, by default it is set to `/tmp/cloudml-samples/molecules`. To change it you can use the `--work-dir` option:

```bash
# It can be a local path
python main.py --work-dir ~/cloudml-samples/molecules

# It can also be a path in Google Cloud Storage
BUCKET=gs://<YOUR_BUCKET_NAME>
python main.py --work-dir $BUCKET/cloudml-samples/molecules
```

This will automatically download the data files, do all the preprocessing and model training. By default it will use 5 data files, each one with 25,000 molecules. To change the number of data files to use, use the `--total-data-files` option:
```bash
# To use 10 files, that would be 250,000 molecules
python main.py --total-data-files 10
```

For more help, you can always use the `--help` option:
```bash
python main.py --help
```

## Running in Google Cloud
To run in Google Cloud, you just have to add the `--cloud` flag. Running in cloud also requires to provide your Project ID via the `--project` option. The `--work-dir` also has to be set to a location in Google Cloud Storage.

```bash
PROJECT=$(gcloud config get-value project)
BUCKET=gs://<YOUR_BUCKET_NAME>

WORK_DIR=$BUCKET/cloudml-samples/molecules

python main.py \
  --cloud \
  --project $PROJECT \
  --work-dir $WORK_DIR \
  --total-data-files 100
```

## Requesting predictions
First, we have to create a model in Cloud ML Engine.
```bash
MODEL=molecules
REGION=us-central1

gcloud ml-engine models create $MODEL \
  --regions $REGION
```

Then, we have to create a version for that model.
```bash
VERSION=v1
MODEL_DIR=$WORK_DIR/model/export/$MODEL/<TIMESTAMP>

gcloud ml-engine versions create $VERSION \
  --model $MODEL \
  --origin $MODEL_DIR
```

Having a model version, we can now send prediction requests via gcloud too. This is a file with one JSON request per line.
```bash
gcloud ml-engine predict \
  --model $MODEL \
  --version $VERSION \
  --json-instances sample_request.json
```

## End-to-end run
```bash
PROJECT=<YOUR_PROJECT_ID>
BUCKET=gs://<YOUR_BUCKET_NAME>

# Settings
WORK_DIR=$BUCKET/cloudml-samples/molecules
MODEL=molecules
VERSION=v1
REGION=us-central1

# Preprocess and train the model
python main.py \
  --cloud \
  --project $PROJECT \
  --model-name $MODEL \
  --work-dir $WORK_DIR \
  --total-data-files 1000

# This is assuming that this is the only exported model available
MODEL_DIR=$(gsutil ls $WORK_DIR/model/export/$MODEL | egrep "$MODEL/[0-9]+")

# Create the model for predictions
gcloud ml-engine models create $MODEL \
  --regions $REGION

# Create the model version for predictions
gcloud ml-engine versions create $VERSION \
  --model $MODEL \
  --origin $MODEL_DIR

# Send the predictions
gcloud ml-engine predict \
  --model $MODEL \
  --version $VERSION \
  --json-instances sample_request.json
```
