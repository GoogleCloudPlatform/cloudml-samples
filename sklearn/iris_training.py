# This file is for training on Cloud ML Engine with scikit-learn.
# TODO(0olwzo0): Remove this file after stopping referring this file from CMLE
# public doc.

# [START setup]
import datetime
import os
import subprocess
import sys
import pandas as pd

from sklearn import svm
from sklearn.externals import joblib

# Fill in your Cloud Storage bucket name
BUCKET_ID = <YOUR_BUCKET_NAME>
# [END setup]


# [START download-data]
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
# [END download-data]


# [START load-into-pandas]
# Load data into pandas
iris_data = pd.read_csv(iris_data_filename).values
iris_target = pd.read_csv(iris_target_filename).values
iris_target = iris_target.reshape((iris_target.size,))
# [END load-into-pandas]


# [START train-and-save-model]
# Train the model
classifier = svm.SVC(verbose=True)
classifier.fit(iris_data, iris_target)

# Export the classifier to a file
model = 'model.joblib'
joblib.dump(classifier, model)
# [END train-and-save-model]


# [START upload-model]
# Upload the saved model file to Cloud Storage
model_path = os.path.join('gs://', bucket, datetime.datetime.now().strftime(
    'iris_%Y%m%d_%H%M%S'), model)
subprocess.check_call(['gsutil', 'cp', model, model_path], stderr=sys.stdout)
# [END upload-model]

