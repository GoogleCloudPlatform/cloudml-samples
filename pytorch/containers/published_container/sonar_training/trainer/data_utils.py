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
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class SonarDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # When iterating through the dataset get the features and targets
        features = self.dataframe.iloc[idx, :-1].values.astype(dtype='float64')

        # Convert the targets to binary values:
        # R = rock --> 0
        # M = mine --> 1
        target = self.dataframe.iloc[idx, -1:].values
        if target[0] == 'R':
            target[0] = 0
        elif target[0] == 'M':
            target[0] = 1
        target = target.astype(dtype='float64')

        # Load the data as a tensor
        data = {'features': torch.from_numpy(features),
                'target': target}
        return data


def download_data():
    """Download the data from Google Cloud Storage"""
    # Load the Dataset from the public GCS bucket
    bucket = storage.Client().bucket('cloud-samples-data')
    # Path to the data inside the public bucket
    blob = bucket.blob('ml-engine/sonar/sonar.all-data')
    # Download the data
    blob.download_to_filename('sonar.all-data')


def load_data(test_split, batch_size):
    """Loads the data"""
    sonar_dataset = SonarDataset('./sonar.all-data')
    # Create indices for the split
    dataset_size = len(sonar_dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = random_split(sonar_dataset,
                                               [train_size, test_size])

    train_loader = DataLoader(
        train_dataset.dataset,
        batch_size=batch_size,
        shuffle=True)
    test_loader = DataLoader(
        test_dataset.dataset,
        batch_size=batch_size,
        shuffle=True)

    return train_loader, test_loader


def save_model(model_dir, model_name):
    """Saves the model to Google Cloud Storage"""
    bucket = storage.Client().bucket(model_dir)
    blob = bucket.blob('{}/{}'.format(
        datetime.datetime.now().strftime('sonar_%Y%m%d_%H%M%S'),
        model_name))
    blob.upload_from_filename(model_name)
