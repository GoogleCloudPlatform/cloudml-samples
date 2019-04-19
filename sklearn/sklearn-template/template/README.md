# Scikit-learn trainer template for AI Platform


This is a template for building a scikit-learn-based machine learning trainer 
that can be run on AI Platform. 

Google Cloud tools used:
- [Google Cloud Platform](https://cloud.google.com/) (GCP) lets you build and 
host applications and websites, store data, and analyze data on Google's 
scalable infrastructure.
- [Cloud ML Engine](https://cloud.google.com/ml-engine/) is a managed service 
that enables you to easily build machine learning models that work on any type 
of data, of any size. This is now part of 
[AI Platform](https://cloud.google.com/ai-platform/).
- [Google Cloud Storage](https://cloud.google.com/storage/) (GCS) is a unified 
object storage for developers and enterprises, from live data serving to data 
analytics/ML to data archiving.
- [Cloud SDK](https://cloud.google.com/sdk/) is a set of tools for Google Cloud 
Platform, which contains e.g. gcloud, gsutil, and bq command-line tools to 
interact with Google Cloud products and services.
- [Google BigQuery](https://cloud.google.com/bigquery/) A fast, highly scalable, 
cost-effective, and fully managed cloud data warehouse for analytics, with even 
built-in machine learning.

## Template structure
```
template 
    |__ config
        |__ config.yaml             # for running normal training job on AI Platform
        |__ hptuning_config.yaml    # for running hyperparameter tunning job on AI Platform    
    |__ scripts
        |__ train.sh                # convenience script for running machine learning training jobs
        |__ deploy.sh               # convenience script for deploying trained scikit-learn model
        |__ predict.sh              # convenience script for requesting online prediction
        |__ predict.py              # helper function for requesting online prediction using python
    |__ trainer                     # trainer package
        |__ metadata.py             # dataset metadata and feature columns definitions
        |__ model.py                # pre-processing and machine learning model pipeline definition
        |__ utils.py                # utility functions including e.g. loading data from bigquery and cloud storage
        |__ task.py                 # training job entry point, handling the parameters passed from command line 
    |__ setup.py                    # specify necessary dependency for running job on AI Platform
    |__ requirements.txt            # specify necessary dependency, helper for setup environemnt for local development
```

## Using the template
### Step 0. Prerequisites
Before you follow the instructions below to adapt the tempate to your machine learning job, 
you need a Google cloud project if you don't have one. You can find detailed instructions 
[here](https://cloud.google.com/dataproc/docs/guides/setup-project).

- Make sure the following API & Services are enabled.
    * Cloud Storage
    * Cloud Machine Learning Engine
    * BigQuery API
    * Cloud Build API (for CI/CD integration)
    * Cloud Source Repositories API (for CI/CD integration)

- Configure project id and bucket id as environment variable.
  ```bash
  $ export PROJECT_ID=[your-google-project-id]
  $ export BUCKET_ID=[your-google-cloud-storage-bucket-name]
  ```
  
- Set up a service account for calls to GCP APIs.  
  More information on setting up a service account can be found 
  [here](https://cloud.google.com/docs/authentication/getting-started).
  
### Step 1. Tailor the scikit-learn trainer to your data

`metadata.py` is where the dataset's metadata is defined. 
By default, the file is configured to train on the Iris dataset, which can be found at 
`bigquery-public-data.ml_datasets.iris`.

```python
# Example for iris dataset
CSV_COLUMNS = None  # Schema of the data. Necessary for data stored in GCS

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

In most cases, only the following items need to be modified, in order to adapt to the target dataset. 
- **COLUMNS**: the schema of ths data, only required for data stored in GCS
- **NUMERIC_FEATURES**: columns those will be treated as numerical features
- **CATEGORICAL_FEATURES**: columns those will be treated as categorical features
- **LABEL**: column that will be treated as label

### Step 2. Modify YAML config files for training on AI Platform
The files are located in `config`:
- `config.yaml`: for running normal training job on AI Platform.
- `hptuning_config.yaml`: for running hyperparameter tuning job on AI Platform.

The YAML files share some configuration parameters. In particular, `runtimeVersion` and `pythonVersion` should
correspond in both files.

```yaml
trainingInput:
  scaleTier: STANDARD_1   # Machine type
  runtimeVersion: "1.13"  # Scikit-learn version
  pythonVersion: "3.5"    # only support python 2.7 and 3.5
```

More information on supported runtime version can be found 
[here](https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list).

### Step 3. Submit scikit-learn training job

You can run ML training jobs through the `train.sh` Bash script.

```shell
bash scripts/train.sh [INPUT] [RUN_ENV] [RUN_TYPE] [EXTRA_TRAINER_ARGS]
```
- INPUT: Dataset to use for training and evaluation, which can be BigQuery table or a file (CSV).
         BigQuery table should be specified as `PROJECT_ID.DATASET.TABLE_NAME`.
- RUN_ENV: (Optional), whether to run `local` (on-prem) or `remote` (GCP). Default value is `local`.
- RUN_TYPE: (Optional), whether to run `train` or `hptuning`. Default value is `train`.
- EXTRA_TRAINER_ARGS: (Optional), additional arguments to pass to the trainer.

**Note**: Please make sure the REGION is set to a supported Cloud region for your project in `train.sh`
```shell
REGION=us-central1
```

### Step 4. Deploy the trained model

The trained model can then be deployed to AI Platform for online serving using the `deploy.sh` script.

```shell
bash scripts/deploy.sh [MODEL_DIR] [MODEL_NAME] [VERSION_NAME]
```

where:

- MODEL_DIR: Path to directory containing trained and exported scikit-learn model.
- MODEL_NAME: Name of the model to be deployed.
- VERSION_NAME: Version of the model to be deployed`.

**Note**: Please make sure the following parameters are properly set in deploy.sh 
```shell
REGION=us-central1

# The following two parameters should be aligned with those used during
# training job, i.e., specified in the yaml files under config/
RUN_TIME=1.13
PYTHON_VERSION=3.5 # only support python 2.7 and 3.5
```

### Step 5. Run predictions using the deployed model

After the model is successfully deployed, you can send small samples of new data to the API associated with the model,
and it would return predictions in the response. 
There are two helper scripts available, `predict.sh` and `predict.py`, which use gcloud and Python API for 
requesting predictions respectively.

```shell
bash scripts/predict.sh [INPUT_DATA_FILE] [MODEL_NAME] [VERSION_NAME]
```

where:

- INPUT_DATA_FILE: Path to sample file contained data in line-delimited JSON format. 
  See `sample_data/sample.txt` for an example. More information can be found 
  [here](https://cloud.google.com/ml-engine/docs/scikit/online-predict#formatting_instances_as_lists).
- MODEL_NAME: Name of the deployed model to use.
- VERSION_NAME: Version of the deployed model to use.
