#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import metadata

import featurizer
import task


# ****************************************************************************************
# YOU MAY MODIFY THESE FUNCTIONS TO USE DIFFERENT ESTIMATORS OR CONFIGURE THE CURRENT ONES
# ****************************************************************************************


def metric_fn(labels, predictions):
    """ Defines extra evaluation metrics to canned and custom estimators.
    By default, this returns an empty dictionary

    Args:
        labels: A Tensor of the same shape as predictions
        predictions: A Tensor of arbitrary shape
    Returns:
        dictionary of string:metric
    """
    metrics = {}

    pred_values = predictions['predictions']

    metrics["mae"] = tf.metrics.mean_absolute_error(labels, pred_values)
    metrics["rmse"] = tf.metrics.root_mean_squared_error(labels, pred_values)

    return metrics


def create_regressor(config):
    """ Create a DNNLinearCombinedRegressor based on the HYPER_PARAMS in task.py

    Args:
        config - used for model directory
    Returns:
        DNNLinearCombinedRegressor
    """

    feature_columns = list(featurizer.create_feature_columns().values())

    deep_columns, wide_columns = featurizer.get_deep_and_wide_columns(
        feature_columns
    )


    # Change the optimisers for the wide and deep parts of the model if you wish
    linear_optimizer = tf.train.FtrlOptimizer(learning_rate=task.HYPER_PARAMS.learning_rate)
    dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=task.HYPER_PARAMS.learning_rate)

    estimator = tf.estimator.DNNLinearCombinedRegressor(

        linear_optimizer=linear_optimizer,
        linear_feature_columns=wide_columns,

        dnn_feature_columns=deep_columns,
        dnn_optimizer=dnn_optimizer,

        weight_column=metadata.WEIGHT_COLUMN_NAME,

        dnn_hidden_units=construct_hidden_units(),
        dnn_activation_fn=tf.nn.relu,
        dnn_dropout=task.HYPER_PARAMS.dropout_prob,

        config=config,
    )

    estimator = tf.contrib.estimator.add_metrics(estimator, metric_fn)

    print("creating a regression model: {}".format(estimator))

    return estimator


# ***************************************************************************************
# YOU NEED TO MODIFY THIS FUNCTIONS IF YOU WANT TO IMPLEMENT A CUSTOM ESTIMATOR
# ***************************************************************************************


def create_estimator(config):
    """ Create a custom estimator based on _model_fn

    Args:
        config - used for model directory
    Returns:
        Estimator
    """

    def _inference(features):
        """ Create the model structure and compute the logits """

        hidden_units = construct_hidden_units()
        output_layer_size = 1  # because it is a regression problem

        feature_columns = list(featurizer.create_feature_columns().values())

        # create the deep columns: dense + indicators
        deep_columns, _ = featurizer.get_deep_and_wide_columns(
            feature_columns
        )

        # Create input layer based on features
        input_layer = tf.feature_column.input_layer(features=features,
                                                    feature_columns=deep_columns)

        # Create hidden layers (cnn, rnn, dropouts, etc.) given the input layer
        hidden_layers = tf.contrib.layers.stack(inputs=input_layer,
                                                layer=tf.contrib.layers.fully_connected,
                                                stack_args=hidden_units,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

        # Create output (logits) layer given the hidden layers (usually without applying any activation functions)
        logits = tf.layers.dense(inputs=hidden_layers,
                                 units=output_layer_size,
                                 activation=None)

        return logits

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""

        # Update learning rate using exponential decay method
        current_learning_rate = update_learning_rate()

        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(
            learning_rate=current_learning_rate)

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        return train_op

    def _model_fn(features, labels, mode):
        """ Create the model structure and compute the logits """

        logits = _inference(features)

        head = tf.contrib.estimator.regression_head(
            weight_column=metadata.WEIGHT_COLUMN_NAME,
            label_dimension=1,
            # loss_reduction=tf.losses.Reduction.SUM, # tf v1.6
            # loss_fn=_loss_fn, # tf v1.6
            name='regression_head'
        )

        return head.create_estimator_spec(
            features,
            mode,
            logits,
            labels=labels,
            train_op_fn=_train_op_fn
        )

    print("creating a custom model...")

    estimator = tf.estimator.Estimator(model_fn=_model_fn, config=config)

    estimator = tf.contrib.estimator.add_metrics(estimator, metric_fn)

    return estimator


# ***************************************************************************************
# HELPER FUNCTIONS USED FOR CONSTRUCTING THE MODELS
# ***************************************************************************************


def construct_hidden_units():
    """ Create the number of hidden units in each layer

    if the HYPER_PARAMS.layer_sizes_scale_factor > 0 then it will use a "decay" mechanism
    to define the number of units in each layer. Otherwise, task.HYPER_PARAMS.hidden_units
    will be used as-is.

    Returns:
        list of int
    """
    hidden_units = list(map(int, task.HYPER_PARAMS.hidden_units.split(',')))

    if task.HYPER_PARAMS.layer_sizes_scale_factor > 0:
        first_layer_size = hidden_units[0]
        scale_factor = task.HYPER_PARAMS.layer_sizes_scale_factor
        num_layers = task.HYPER_PARAMS.num_layers

        hidden_units = [
            max(2, int(first_layer_size * scale_factor ** i))
            for i in range(num_layers)
        ]

    print("Hidden units structure: {}".format(hidden_units))

    return hidden_units


def update_learning_rate():
    """ Updates learning rate using an exponential decay method

    Returns:
       float - updated (decayed) learning rate
    """
    initial_learning_rate = task.HYPER_PARAMS.learning_rate
    decay_steps = task.HYPER_PARAMS.train_steps  # decay after each training step
    decay_factor = task.HYPER_PARAMS.learning_rate_decay_factor  # if set to 1, then no decay.

    global_step = tf.train.get_global_step()

    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_factor)

    return learning_rate
