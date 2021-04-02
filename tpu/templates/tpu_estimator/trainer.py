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


import argparse
import numpy as np
import tensorflow as tf


# ## The model function
# There are two differences in the model function when using TPUs:
# 
# * The optimizer needs to be wrapped in a `tf.contrib.tpu.CrossShardOptimizer`.
#
# * The model function should return a `tf.contrib.tpu.TPUEstimatorSpec`.
#


def model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()
    hidden = tf.layers.dense(features, 10, activation=tf.nn.relu)
    output = tf.layers.dense(hidden, 1)

    predictions = output
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        loss = tf.nn.l2_loss(predictions - labels)

        # define train_op
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)

        # wrapper to make the optimizer work with TPUs
        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=global_step)

    if params['use_tpu']:
        # TPU version of EstimatorSpec
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


# ## The input function
# tf.data.Dataset is the best choice for building the input function.
# Even though datasets can determine the shape of the data at *runtime*,
# TPUs need to know the shape of the tensors *when the graph is built*.
# This typically means two things:
#
# * Set `drop_remainder=True` in the `dataset.batch` call.
#
# * Set tensor shapes to make sure the features and labels do not have any unknown dimensions.
#


def train_input_fn(params={}):
    # make some fake regression data
    x = np.random.rand(100, 5)
    w = np.random.rand(5)
    y = np.sum(x * w, axis=1)

    # TPUs currently do not support float64
    x_tensor = tf.constant(x, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.float32)

    # create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

    # TPUEstimator passes params when calling input_fn
    batch_size = params.get('batch_size', 16)

    dataset = dataset.repeat().shuffle(32).batch(batch_size, drop_remainder=True)

    # TPUs need to know all dimensions when the graph is built
    # Datasets know the batch size only when the graph is run
    def set_shapes(features, labels):
        features_shape = features.get_shape().merge_with([batch_size, None])
        labels_shape = labels.get_shape().merge_with([batch_size])

        features.set_shape(features_shape)
        labels.set_shape(labels_shape)

        return features, labels

    dataset = dataset.map(set_shapes)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


# ## The TPUEstimator
# The TPUEstimator is similar to the usual Estimator, but requires a
# slightly different run_config, since it needs to know where to connect
# to the TPU workers.
#
# This is done through `tf.contrib.cluster_resolver.TPUClusterResolver`,
# which is passed into a `tf.contrib.tpu.TPUConfig`, which in turn is
# passed into `tf.contrib.tpu.RunConfig`.
#


def main(args):
    # pass the args as params so the model_fn can use
    # the TPU specific args
    params = vars(args)

    if args.use_tpu:
        # additional configs required for using TPUs
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu)
        tpu_config = tf.contrib.tpu.TPUConfig(
            num_shards=8, # using Cloud TPU v2-8
            iterations_per_loop=args.save_checkpoints_steps)

        # use the TPU version of RunConfig
        config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=args.model_dir,
            tpu_config=tpu_config,
            save_checkpoints_steps=args.save_checkpoints_steps,
            save_summary_steps=100)

        # TPUEstimator
        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            config=config,
            params=params,
            train_batch_size=args.train_batch_size,
            eval_batch_size=32,
            export_to_tpu=False)
    else:
        config = tf.estimator.RunConfig(model_dir=args.model_dir)

        estimator = tf.estimator.Estimator(
            model_fn,
            config=config,
            params=params)

    estimator.train(train_input_fn, max_steps=args.max_steps)


# ## Training
# Depending on where the training job is run, the `TPUClusterResolver`
# needs different input to access the TPU workers:
#
# * On AI Platform: the input should be `None` 
#   and the service will handle it.
#
# * On Compute Engine: the input should be the name of TPU you create
#   before starting the training job.
#
# * On Colab: the input should be the grpc URI from the environment
# variable `COLAB_TPU_ADDR`; the Colab runtime type should be set to
# TPU for this environment variable to be automatically set.
#  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tpu-template',
        help='Location to write checkpoints and summaries to.  Must be a GCS URI when using Cloud TPU.')
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000,
        help='The total number of steps to train the model.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=16,
        help='The training batch size.  The training batch is divided evenly across the TPU cores.')
    parser.add_argument(
        '--save-checkpoints-steps',
        type=int,
        default=100,
        help='The number of training steps before saving each checkpoint.')
    parser.add_argument(
        '--use-tpu',
        action='store_true',
        help='Whether to use TPU.')
    parser.add_argument(
        '--tpu',
        default=None,
        help='The name or GRPC URL of the TPU node.  Leave it as `None` when training on AI Platform.')

    args, _ = parser.parse_known_args()

    main(args)
