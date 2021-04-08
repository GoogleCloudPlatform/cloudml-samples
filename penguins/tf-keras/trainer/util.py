# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to download and preprocess the Penguin data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

# Keras specific
from tensorflow.keras.utils import to_categorical

# Storage directory
DATA_DIR = os.path.join(tempfile.gettempdir(), "penguins_data")

# Download options.
DATA_URL = "https://storage.googleapis.com/cloud-samples-data/ai-platform/penguins"  # noqa
TRAINING_FILE = "penguins.data.csv"
EVAL_FILE = "penguins.test.csv"
TRAINING_URL = "%s/%s" % (DATA_URL, TRAINING_FILE)
EVAL_URL = "%s/%s" % (DATA_URL, EVAL_FILE)

# These are the features in the dataset.
# Dataset information: https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data  # noqa
_CSV_COLUMNS = [
    "species",
    "island",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]

# This is the label (target) we want to predict.
_LABEL_COLUMN = "species"

UNUSED_COLUMNS = []

_CATEGORICAL_TYPES = {
    "island": pd.api.types.CategoricalDtype(
        categories=[
            "Torgersen",
            "Biscoe",
            "Dream",
        ]
    ),
    "species": pd.api.types.CategoricalDtype(
        categories=[
            "Adelie Penguin (Pygoscelis adeliae)",
            "Chinstrap penguin (Pygoscelis antarctica)",
            "Gentoo (Pygoscelis papua)",
        ]
    ),
    "sex": pd.api.types.CategoricalDtype(
        categories=[
            "MALE",
            "FEMALE",
        ]
    ),
}


def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format.

    The CSVs may use spaces after the comma delimters (non-standard) or include
    rows which do not represent well-formed examples. This function strips out
    some of these problems.

    Args:
      filename: filename to save url to
      url: URL of resource to download
    """
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.io.gfile.GFile(temp_file, "r") as temp_file_object:
        with tf.io.gfile.GFile(filename, "w") as file_object:
            for line in temp_file_object:
                line = line.strip()
                line = line.replace(", ", ",")
                if not line or "," not in line:
                    continue
                if line[-1] == ".":
                    line = line[:-1]
                line += "\n"
                file_object.write(line)
    tf.io.gfile.remove(temp_file)


def download(data_dir):
    """Downloads data if it is not already present.

    Args:
      data_dir: directory where we will access/save the data
    """
    tf.io.gfile.makedirs(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not tf.io.gfile.exists(training_file_path):
        _download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    if not tf.io.gfile.exists(eval_file_path):
        _download_and_clean_file(eval_file_path, EVAL_URL)

    return training_file_path, eval_file_path


def preprocess(dataframe):
    """Converts categorical features to numeric. Removes unused columns.

    Args:
      dataframe: Pandas dataframe with raw data

    Returns:
      Dataframe with preprocessed data
    """
    dataframe = dataframe.drop(columns=UNUSED_COLUMNS)

    # Drop rows with NaN's
    dataframe = dataframe.dropna()

    # Convert integer valued (numeric) columns to floating point
    numeric_columns = dataframe.select_dtypes(["int32", "float64"]).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].astype("float32")

    # Convert categorical columns to numeric
    cat_columns = dataframe.select_dtypes(["object"]).columns

    dataframe[cat_columns] = dataframe[cat_columns].apply(
        lambda x: x.astype(_CATEGORICAL_TYPES[x.name])
    )
    dataframe[cat_columns] = dataframe[cat_columns].apply(
        lambda x: x.cat.codes)
    return dataframe


def standardize(dataframe):
    """Scales numerical columns using their means and standard deviation to get
    z-scores: the mean of each numerical column becomes 0, and the standard
    deviation becomes 1. This can help the model converge during training.

    Args:
      dataframe: Pandas dataframe

    Returns:
      Input dataframe with the numerical columns scaled to z-scores
    """
    dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
    # Normalize numeric columns.
    for column, dtype in dtypes:
        if dtype == "float32":
            dataframe[column] -= dataframe[column].mean()
            dataframe[column] /= dataframe[column].std()
    return dataframe


def load_data():
    """Loads data into preprocessed (train_x, train_y, eval_y, eval_y)
    dataframes.

    Returns:
      A tuple (train_x, train_y, eval_x, eval_y), where train_x and eval_x are
      Pandas dataframes with features for training and train_y and eval_y are
      numpy arrays with the corresponding labels.
    """
    # Download dataset: Training and eval csv files.
    training_file_path, eval_file_path = download(DATA_DIR)

    # This data uses the value '?' for missing entries. We use
    # na_values to
    # find ? and set it to NaN.
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv
    # .html
    NA_VALUES = ["NA", "."]
    train_df = pd.read_csv(
        training_file_path, names=_CSV_COLUMNS, header=0, na_values=NA_VALUES
    )
    eval_df = pd.read_csv(
        eval_file_path, names=_CSV_COLUMNS, header=0, na_values=NA_VALUES
    )

    train_df = preprocess(train_df)
    eval_df = preprocess(eval_df)

    # Split train and eval data with labels. The pop method copies and removes
    # the label column from the dataframe.
    train_x, train_y = train_df, train_df.pop(_LABEL_COLUMN)
    eval_x, eval_y = eval_df, eval_df.pop(_LABEL_COLUMN)

    # Join train_x and eval_x to normalize on overall means and standard
    # deviations. Then separate them again.
    all_x = pd.concat([train_x, eval_x], keys=["train", "eval"])
    all_x = standardize(all_x)
    train_x, eval_x = all_x.xs("train"), all_x.xs("eval")

    # Reshape label columns for use with tf.data.Dataset
    train_y = np.asarray(train_y).astype("float32").reshape((-1, 1))
    eval_y = np.asarray(eval_y).astype("float32").reshape((-1, 1))

    # one hot encode outputs
    train_y = to_categorical(train_y)
    eval_y = to_categorical(eval_y)

    return train_x, train_y, eval_x, eval_y
