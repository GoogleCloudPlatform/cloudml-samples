# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Defines a Wide + Deep model for classification on structured data.

Tutorial on wide and deep: https://www.tensorflow.org/tutorials/wide_and_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import trainer.featurizer as featurizer


def build_estimator(config, embedding_size=8, hidden_units=None):
    """Build a wide and deep model for predicting income category.

    Wide and deep models use deep neural nets to learn high level abstractions
    about complex features or interactions between such features.
    These models then combined the outputs from the DNN with a linear regression
    performed on simpler features. This provides a balance between power and
    speed that is effective on many structured data problems.

    You can read more about wide and deep models here:
    https://research.googleblog.com/2016/06/wide-deep-learning-better
    -together-with.html

    To define model we can use the prebuilt DNNCombinedLinearClassifier class,
    and need only define the data transformations particular to our dataset, and
    then
    assign these (potentially) transformed features to either the DNN, or linear
    regression portion of the model.

    Args:
      config: (tf.contrib.learn.RunConfig) defining the runtime environment for
        the estimator (including model_dir).
      embedding_size: (int), the number of dimensions used to represent
        categorical features when providing them as inputs to the DNN.
      hidden_units: [int], the layer sizes of the DNN (input layer first)

    Returns:
      A DNNCombinedLinearClassifier
    """

    (deep_columns, wide_columns) = featurizer.get_deep_and_wide_columns(
        embedding_size)

    return tf.estimator.DNNLinearCombinedClassifier(
        config=config,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25])
