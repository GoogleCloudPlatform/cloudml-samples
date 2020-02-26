# [START setup]
import datetime
import os
import subprocess

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from google.cloud import storage
import xgboost as xgb


# TODO: REPLACE 'BUCKET_CREATED_ABOVE' with your GCS BUCKET_ID
BUCKET_ID = 'torryyang-xgb-models'
# [END setup]

# ---------------------------------------
# 1. Add code to download the data from GCS (in this case, using the publicly hosted data).
# AI Platform will then be able to use the data when training your model.
# ---------------------------------------
# [START download-data]
census_data_filename = 'adult.data.csv'

# Public bucket holding the census data
bucket = storage.Client().bucket('cloud-samples-data')

# Path to the data inside the public bucket
data_dir = 'ai-platform/census/data/'

# Download the data
blob = bucket.blob(''.join([data_dir, census_data_filename]))
blob.download_to_filename(census_data_filename)
# [END download-data]

# ---------------------------------------
# This is where your model code would go. Below is an example model using the census dataset.
# ---------------------------------------

# [START define-and-load-data]

# these are the column labels from the census data files
COLUMNS = (
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income-level'
)
# categorical columns contain data that need to be turned into numerical values before being used by XGBoost
CATEGORICAL_COLUMNS = (
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
)

# Load the training census dataset
with open(census_data_filename, 'r') as train_data:
    raw_training_data = pd.read_csv(train_data, header=None, names=COLUMNS)
# remove column we are trying to predict ('income-level') from features list
train_features = raw_training_data.drop('income-level', axis=1)
# create training labels list
train_labels = (raw_training_data['income-level'] == ' >50K')

# [END define-and-load-data]

# [START categorical-feature-conversion]
# Since the census data set has categorical features, we need to convert
# them to numerical values.
# convert data in categorical columns to numerical values
encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}
for col in CATEGORICAL_COLUMNS:
    train_features[col] = encoders[col].fit_transform(train_features[col])
# [END categorical-feature-conversion]

# [START load-into-dmatrix-and-train]
# load data into DMatrix object
dtrain = xgb.DMatrix(train_features, train_labels)
# train model
bst = xgb.train({}, dtrain, 20)
# [END load-into-dmatrix-and-train]

# ---------------------------------------
# 2. Train and save the model to GCS
# ---------------------------------------
# [START export-to-gcs]
# Export the model to a file
model = 'model.bst'
bst.save_model(model)

# Upload the model to GCS
bucket = storage.Client().bucket(BUCKET_ID)
blob = bucket.blob('{}/{}'.format(
    datetime.datetime.now().strftime('census_%Y%m%d_%H%M%S'),
    model))
blob.upload_from_filename(model)
# [END export-to-gcs]
