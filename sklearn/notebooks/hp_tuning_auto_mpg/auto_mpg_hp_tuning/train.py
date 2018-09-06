#!/usr/bin/env python
 # Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START mle_sklearn_hp_tuning_setup]
import argparse
import datetime
import os
import pandas as pd
import subprocess

from google.cloud import storage
import hypertune

from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
# [END mle_sklearn_hp_tuning_setup]

# ---------------------------------------
# 1. Here we load the hyperparameter values that are passed to the model during training.
# ---------------------------------------
# In this tutorial, the Lasso regressor is used, because it has several parameters
# that can be used to help demonstrate how to choose HP tuning values.
# (The range of values are set below in the configuration file for the HP tuning values.)
# [START mle_sklearn_hp_tuning_argparse]
parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-dir',  # handled automatically by ML Engine
    help='GCS location to write checkpoints and export models',
    required=True
)
parser.add_argument(
    '--alpha',  # Specified in the config file
    help='Constant that multiplies the L1 term.',
    default=1.0,
    type=float
)
parser.add_argument(
    '--max_iter',  # Specified in the config file
    help='The maximum number of iterations.',
    default=1000,
    type=int
)
parser.add_argument(
    '--tol',  # Specified in the config file
    help='The tolerance for the optimization: if the updates are smaller than tol, '
         'the optimization code checks the dual gap for optimality and continues '
         'until it is smaller than tol.',
    default=0.0001,
    type=float
)
parser.add_argument(
    '--selection',  # Specified in the config file
    help='Supported criteria are “cyclic” loop over features sequentially and '
         '“random” a random coefficient is updated every iteration ',
    default='cyclic'
)

args = parser.parse_args()
# [END mle_sklearn_hp_tuning_argparse]


# ---------------------------------------
# 2. Add code to download the data from GCS (in this case, using the publicly hosted data).
#    ML Engine will then be able to use the data when training your model.
# ---------------------------------------
# [START mle_sklearn_hp_tuning_download_data]
# Public bucket holding the auto mpg data
bucket = storage.Client().bucket('cloud-samples-data')
# Path to the data inside the public bucket
blob = bucket.blob('ml-engine/sklearn/auto_mpg_data/auto-mpg.data')
# Download the data
blob.download_to_filename('auto-mpg.data')
# [END mle_sklearn_hp_tuning_download_data]


# ---------------------------------------
# This is where your model code would go. Below is an example model using the auto mpg dataset.
# ---------------------------------------
# [START mle_sklearn_hp_tuning_define_and_load_data]
# Define the format of your input data including unused columns
# (These are the columns from the auto-mpg data files)
COLUMNS = (
    'mpg',
    'cylinders',
    'displacement',
    'horsepower',
    'weight',
    'acceleration',
    'model-year',
    'origin',
    'car-name'
)

# Load the training auto mpg dataset
with open('./auto-mpg.data', 'r') as train_data:
    raw_training_data = pd.read_csv(train_data, header=None, names=COLUMNS, delim_whitespace=True)

# Remove the column we are trying to predict ('mpg') from our features list
# Convert the Dataframe to a lists of lists
features = raw_training_data.drop('mpg', axis=1).drop('car-name', axis=1).values.tolist()

# Create our training labels list, convert the Dataframe to a lists of lists
labels = raw_training_data['mpg'].values.tolist()

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.15)
# [END mle_sklearn_hp_tuning_define_and_load_data]

# ---------------------------------------
# 3. Use the value passed in those arguments to set the corresponding hyperparameters
#    in your application's scikit-learn code.
# ---------------------------------------
# [START mle_sklearn_hp_tuning_create_model]
# Create the regressor, here we will use a Lasso Regressor to demonstrate the use of HP Tuning.
# Here is where we set the variables used during HP Tuning from
# the parameters passed into the python script
regressor = Lasso(
    alpha=args.alpha,
    max_iter=args.max_iter,
    tol=args.tol,
    selection=args.selection)

# Transform the features and fit them to the regressor
regressor.fit(train_features, train_labels)
# [END mle_sklearn_hp_tuning_create_model]

# ---------------------------------------
# 4. Report the mean accuracy as hyperparameter tuning objective metric.
# ---------------------------------------
# [START mle_sklearn_hp_tuning_metrics]
# Calculate the mean accuracy on the given test data and labels.
score = regressor.score(test_features, test_labels)

# The default name of the metric is training/hptuning/metric. 
# We recommend that you assign a custom name. The only functional difference is that 
# if you use a custom name, you must set the hyperparameterMetricTag value in the 
# HyperparameterSpec object in your job request to match your chosen name.
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#HyperparameterSpec
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='my_metric_tag',
    metric_value=score,
    global_step=1000)
# [END mle_sklearn_hp_tuning_metrics]

# ---------------------------------------
# 5. Export and save the model to GCS
# ---------------------------------------
# [START mle_sklearn_hp_export_to_gcs]
# Export the model to a file
model_filename = 'model.joblib'
joblib.dump(regressor, model_filename)

# Example: job_dir = 'gs://BUCKET_ID/scikit_learn_job_dir/1'
job_dir =  args.job_dir.replace('gs://', '')  # Remove the 'gs://'
# Get the Bucket Id
bucket_id = job_dir.split('/')[0]
# Get the path
bucket_path = job_dir.lstrip('{}/'.format(bucket_id))  # Example: 'scikit_learn_job_dir/1'

# Upload the model to GCS
bucket = storage.Client().bucket(bucket_id)
blob = bucket.blob('{}/{}'.format(
    bucket_path,
    model_filename))
blob.upload_from_filename(model_filename)
# [END mle_sklearn_hp_export_to_gcs]