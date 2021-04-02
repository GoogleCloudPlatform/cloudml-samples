# This file is for training on AI Platform with XGBoost.


# [START setup]
import datetime
import os
import subprocess
import sys
import pandas as pd
import xgboost as xgb

# Fill in your Cloud Storage bucket name
BUCKET_NAME = '<YOUR_BUCKET_NAME>'
# [END setup]

# [START download-data]
iris_data_filename = 'iris_data.csv'
iris_target_filename = 'iris_target.csv'
data_dir = 'gs://cloud-samples-data/ai-platform/iris'

# gsutil outputs everything to stderr so we need to divert it to stdout.
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_data_filename),
                       iris_data_filename], stderr=sys.stdout)
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_target_filename),
                       iris_target_filename], stderr=sys.stdout)
# [END download-data]


# [START load-into-pandas]
# Load data into pandas, then use `.values` to get NumPy arrays
iris_data = pd.read_csv(iris_data_filename).values
iris_target = pd.read_csv(iris_target_filename).values

# Convert one-column 2D array into 1D array for use with XGBoost
iris_target = iris_target.reshape((iris_target.size,))
# [END load-into-pandas]


# [START train-and-save-model]
# Load data into DMatrix object
dtrain = xgb.DMatrix(iris_data, label=iris_target)

# Train XGBoost model
bst = xgb.train({}, dtrain, 20)

# Export the classifier to a file
model_filename = 'model.bst'
bst.save_model(model_filename)
# [END train-and-save-model]


# [START upload-model]
# Upload the saved model file to Cloud Storage
gcs_model_path = os.path.join('gs://', BUCKET_NAME,
    datetime.datetime.now().strftime('iris_%Y%m%d_%H%M%S'), model_filename)
subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
    stderr=sys.stdout)
# [END upload-model]
