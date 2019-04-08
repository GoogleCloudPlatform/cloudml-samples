# Template for training a Scikit-learn Model on Cloud ML Engine


This template is designed for building a scikit-learn-based machine learning trainer that can be run on 
Cloud ML Engine (CMLE) at scale. Before you jump in, let’s cover some of the different tools you’ll be using to get 
your job running on CMLE.

- [Google Cloud Platform](https://cloud.google.com/) (GCP) lets you build and host applications and websites, store data, 
and analyze data on Google's scalable infrastructure.
- [Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/) (CMLE) is a managed service that enables you to 
easily build machine learning models that work on any type of data, of any size.
- [Google Cloud Storage](https://cloud.google.com/storage/) (GCS) is a unified object storage for developers and 
enterprises, from live data serving to data analytics/ML to data archiving.
- [Cloud SDK](https://cloud.google.com/sdk/) is a set of tools for Google Cloud Platform, which contains e.g. gcloud, 
gsutil, and bq command-line tools to interact with Google Cloud products and services.
- [Google BigQuery](https://cloud.google.com/bigquery/) A fast, highly scalable, cost-effective, and fully managed 
cloud data warehouse for analytics, with even built-in machine learning.

# Structure of the template
```
Template              
    |__ scripts
        |__ train.sh            # convenience script for running machine learning training jobs 
    |__ trainer                 # trainer package
        |__ metadata.py         # dataset metadata and feature columns definitions
        |__ model.py            # pre-processing and machine learning model pipeline definition
        |__ utils.py            # utility functions including e.g. loading data from bigquery and cloud storage
        |__ task.py             # training job entry point, handling the parameters passed from command line 
    |__ config.yaml             # for running normal training job on CMLE
    |__ hptuning_config.yaml    # for running hyperparameter tunning job on CMLE
    |__ setup.py                # specify necessary dependency for running job on CMLE
    |__ requirements.txt        # specify necessary dependency, helper for setup environemnt for local development
    |__ Dockerfile
```

# Steps to make use of the template
## Step 0. Prerequisite
Before you follow the instructions below to adapt the tempate to your machine learning job, 
you need a Google cloud project if you don't have one. You can find detailed instructions 
[here](https://cloud.google.com/dataproc/docs/guides/setup-project).

- Make sure the following API & Services are enabled.
    * Cloud Storage
    * Cloud Machine Learning Engine
    * BigQuery API
    * Cloud Build API (for CI/CD integration)
    * Cloud Source Repositories API (for CI/CD integration)

- Configure project id as environment variable
  ```bash
  $ export PROJECT_ID=[your-google-project-id]
  ```
  
## Step 1. Modify metadata.py
```python
# Example for iris dataset
COLUMNS = None  # Schema of the data. Necessary for data stored in GCS

NUMERIC_FEATURES = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
]

# Fill this with any categorical features in the dataset
CATEGORICAL_FEATURES = [

] # For iris dataset, there is no categorical feature

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABEL = 'species'

METRIC_FILE_NAME_PREFIX = 'metric'
MODEL_FILE_NAME_PREFIX = 'model'
MODEL_FILE_NAME_SUFFIX = '.joblib'

BASE_QUERY = '''
    Select * From `{table}`
  '''
```

metadata.py is where the dataset's metadata is defined. The code snippets above is an example configured 
for iris dataset (can be found at `bigquery-public-data.ml_datasets.iris`). In most cases, only the following
items need to be modified, in order to adapt to the target dataset. 
- **COLUMNS**: the schema of ths data, only required for data stored in GCS
- **NUMERIC_FEATURES**: columns those will be treated as numerical features
- **CATEGORICAL_FEATURES**: columns those will be treated as categorical features
- **LABEL**: column that will be treated as label

## Step 2. Modify yaml file
There are two yaml files, where
- config.yaml: for running normal training job on CMLE
- hptuning_config.yaml: for running hyperparameter tunning job on CMLE

There is a common portion in both of the yaml file defining critical configurations for training ML model on CMLE. The 
code snippets is an example. In particular, the runtimeVersion and scikit-learn version correspondence 
can be check [here](https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list).
```yaml
trainingInput:
  scaleTier: STANDARD_1   # Machine type
  region: "us-central1"   # GCP region
  runtimeVersion: "1.13"  # Scikit-learn version
  pythonVersion: "2.7"    # Note: Python 3 is also supported
```

## Step 3. Submit ML training job
```shell
bash scripts/train.sh [INPUT_PATH] [RUN_ENV] [RUN_TYPE]
```
- INPUT_PATH: Dataset to use for training and evaluation. Can be BigQuery table or a file (CSV).
              BigQuery table should be specified as `PROJECT_ID.DATASET.TABLE_NAME`.
- RUN_ENV: (Optional), whether to run `local` (on-prem) or `remote` (GCP). Default value is `local`.
- RUN_TYPE: (Optional), whether to run `train` or `hptuning`. Default value is `train`.

## Step 3. Deploy the trained scikit-learn model
After training finishes, the model will be exported to specified job directory in Google Cloud Storage. 
The exported model can then be deployed to CMLE for online serving, the details of which can be 
found [here](https://cloud.google.com/ml-engine/docs/scikit/using-pipelines#store-your-model)

## Optional Step. Cloud Build and CI/CD