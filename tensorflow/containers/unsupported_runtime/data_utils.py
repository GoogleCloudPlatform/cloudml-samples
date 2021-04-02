# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from google.cloud import storage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_data(args):
    """Loads the data"""
    features = pd.read_csv('./sonar.all-data', header=None)
    labels = features[60].values
    features = features.drop(columns=60).values

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)

    train_f, test_f, train_l, test_l = train_test_split(
        features, labels, test_size=args.test_split, random_state=args.seed)
    return train_f, test_f, train_l, test_l


def save_model(model_dir, model_name):
    """Saves the model to Google Cloud Storage"""
    bucket = storage.Client().bucket(model_dir)
    blob = bucket.blob('{}/{}'.format(
        datetime.datetime.now().strftime('sonar_%Y%m%d_%H%M%S'),
        model_name))
    blob.upload_from_filename(model_name)
