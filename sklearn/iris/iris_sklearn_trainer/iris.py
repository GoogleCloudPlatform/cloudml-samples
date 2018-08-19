# This file is for training on Cloud ML Engine with scikit-learn.

# [START mle_iris_training_setup]
import argparse
import datetime
import os
import subprocess
import sys
import pandas as pd

from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True
)
parser.add_argument(
    '--kernel',
    help='Specifies the kernel type to be used in the '
         'algorithm. It must be one of `linear`, `poly`, '
         '`rbf`, `sigmoid` or a callable',
    default='rbf'
)
parser.add_argument(
    '--c',
    help='Penalty parameter C of the error term',
    type=float,
    default='1.0'
)

args = parser.parse_args()
# [END mle_iris_training_setup]

# [START mle_iris_training_download-data]
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
# [END mle_iris_training_download-data]

# [START mle_iris_training_load-data]
# Load data into pandas
iris_data = pd.read_csv(iris_data_filename).values
iris_target = pd.read_csv(iris_target_filename).values
iris_target = iris_target.reshape((iris_target.size,))
# Split the input data into training and test sets.
# A fixed random_state will always return the same split result.
train_data, test_data, train_target, test_target = train_test_split(
    iris_data, iris_target, test_size=0.4, random_state=0)
# [END mle_iris_training_load-data]

# [START mle_iris_training_train_save_model]
# Train the model
classifier = svm.SVC(verbose=True, kernel=args.kernel, C=args.c)
classifier.fit(train_data, train_target)
# Export the classifier to a file
model = 'model.joblib'
joblib.dump(classifier, model)
# [END mle_iris_training_train_save_model]

# [START mle_iris_training_hp_tuning_metrics]
# Calculate the mean accuracy on the given test data and labels.
score = classifier.score(test_data, test_target)
# Report the mean accuracy as hyperparameter tuning objective metric.
import hypertune
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='my_metric_tag',
    metric_value=score,
    global_step=1)
# [END mle_iris_training_hp_tuning_metrics]

# [START mle_iris_training_upload_model]
# Upload the saved model file to Cloud Storage
model_path = os.path.join(args.job_dir, model)
subprocess.check_call(['gsutil', 'cp', model, model_path], stderr=sys.stdout)
# [END mle_iris_training_upload_model]

