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

# This file is for training on Cloud ML Engine with XGBoost.

# [START mle-iris-setup]
import argparse
import datetime
import os
import subprocess
import sys
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True,
)

parser.add_argument(
    '--num-boost-round',
    help='Number of boosting iterations.',
    type=int,
    default=10
)

args = parser.parse_args()
# [END mle-iris-setup]

# [START mle-iris-download-data]
iris_data_filename = 'iris_data.csv'
iris_target_filename = 'iris_target.csv'
data_dir = 'gs://cloud-samples-data/ml-engine/iris'

# gsutil outputs everything to stderr so we need to divert it to stdout.
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_data_filename),
                       iris_data_filename], stderr=sys.stdout)
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_target_filename),
                       iris_target_filename], stderr=sys.stdout)
# [END mle-iris-download-data]

# [START mle-iris-load-into-pandas]
# Load data into pandas
iris_data = pd.read_csv(iris_data_filename).values
iris_target = pd.read_csv(iris_target_filename).values
iris_target = iris_target.reshape((iris_target.size,))
# Split the input data into training and test sets.
# A fixed random_state will always return the same split result.
train_data, test_data, train_target, test_target = train_test_split(
    iris_data, iris_target, test_size=0.4, random_state=0)
# [END mle-iris-load-into-pandas]


# [START mle-iris-train-and-save-model]
# Load data into DMatrix object
dtrain = xgb.DMatrix(train_data, label=train_target)

# Train XGBoost model
bst = xgb.train(params={}, dtrain=dtrain, num_boost_round=args.num_boost_round)

# Export the classifier to a file
model = 'model.bst'
bst.save_model(model)
# [END mle-iris-train-and-save-model]

# [START mle_iris_training_hp_tuning_metrics]
# Calculate the RMSE on the given test data and labels.
deval = xgb.DMatrix(test_data, label=test_target)
rmse = float(bst.eval(deval).split(':')[1])
# Report the mean accuracy as hyperparameter tuning objective metric.
import hypertune
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='my_metric_tag',
    metric_value=rmse,
    global_step=1)
# [END mle_iris_training_hp_tuning_metrics]

# [START mle-iris-upload-model]
# Upload the saved model file to Cloud Storage
model_path = os.path.join(args.job_dir, model)
subprocess.check_call(['gsutil', 'cp', model, model_path], stderr=sys.stdout)
# [END mle-iris-upload-model]
