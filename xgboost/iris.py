# This file is for training on Cloud ML Engine with XGBoost.

import datetime
import os
import subprocess
import sys
import pandas as pd
import xgboost as xgb

# Fill in your Cloud Storage bucket name
BUCKET_ID = <YOUR_BUCKET_NAME>
iris_data_filename = 'iris_data.csv'
iris_target_filename = 'iris_target.csv'
data_dir = os.path.join('gs://cloud-samples-data/ml-engine/iris')

# gsutil outputs everything to stderr so we need to divert it to stdout.
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_data_filename),
                       iris_data_filename], stderr=sys.stdout)
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_target_filename),
                       iris_target_filename], stderr=sys.stdout)

# Load data into pandas
iris_data = pd.read_csv(iris_data_filename).values
iris_target = pd.read_csv(iris_target_filename).values
iris_target = iris_target.reshape((iris_target.size,))

# Load data into DMatrix object
dtrain = xgb.DMatrix(iris_data, label=iris_target)

# Train XGBoost model
bst = xgb.train({}, dtrain, 20)

# Export the classifier to a file
model = 'model.bst'
bst.save_model(model)

# Upload the saved model file to Cloud Storage
model_path = os.path.join('gs://', bucket, datetime.datetime.now().strftime(
    'iris_%Y%m%d_%H%M%S'), model)
subprocess.check_call(['gsutil', 'cp', model, model_path], stderr=sys.stdout)
