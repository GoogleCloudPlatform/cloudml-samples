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
from tensorflow.contrib import summary

N_CLASSES = 10


# ## Gradient Reversal Layer
#
# When applied to a tensor this layer is the identity map, but it reverses
# the sign of the gradient, and optionally multiplies the reversed gradient
# by a weight.
#
# For details, see [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818).
#

class GradientReversalLayer(tf.layers.Layer):
    def __init__(self, weight=1.0):
        super(GradientReversalLayer, self).__init__()
        self.weight = weight

    def call(self, input_):
        @tf.custom_gradient
        def _call(input_):
            def reversed_gradient(output_grads):
                return self.weight * tf.negative(output_grads)

            return input_, reversed_gradient
        
        return _call(input_)


# ## The model function
# The network consists of 3 sub-networks:
#
# * Feature extractor: extracts internal representation for both the source and target distributions.
#
# * Label predictor: predicts label from the extracted features.
#
# * Domain classifier: classifies the origin (`source` or `target`) of the extracted features.
#
#
# Both the label predictor and the domain classifier will try to minimize 
# classification loss, but the gradients backpropagated from the domain
# classifier to the feature extractor have their signs reversed.
#
#
# This model function also shows how to use `host_call` to output summaries.
#

def model_fn(features, labels, mode, params):
    source = features['source']
    target = features['target']
    onehot_labels = tf.one_hot(labels, N_CLASSES)

    global_step = tf.train.get_global_step()

    # In this sample we use dense layers for each of the sub-networks.
    feature_extractor = tf.layers.Dense(7, activation=tf.nn.sigmoid)

    label_predictor_logits = tf.layers.Dense(N_CLASSES)

    # There are two domains, 0: source and 1: target
    domain_classifier_logits = tf.layers.Dense(2)

    source_features = feature_extractor(source)
    target_features = feature_extractor(target)

    # Apply the gradient reversal layer to target features
    gr_weight = params['gr_weight']
    gradient_reversal = GradientReversalLayer(gr_weight)
    target_features = gradient_reversal(target_features)

    # The predictions are the predicted labels from the `target` distribution.
    predictions = tf.nn.softmax(label_predictor_logits(target_features))
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        label_prediction_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=label_predictor_logits(source_features)
        )

        # There are two domains, 0: source and 1: target
        domain_labels = tf.concat((tf.zeros(source.shape[0], dtype=tf.int32), tf.ones(target.shape[0], dtype=tf.int32)), axis=0)
        domain_onehot_labels = tf.one_hot(domain_labels, 2)

        source_target_features = tf.concat([source_features, target_features], axis=0)

        domain_classification_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=domain_onehot_labels,
            logits=domain_classifier_logits(source_target_features)
        )

        lambda_ = params['lambda']

        loss = label_prediction_loss + lambda_ * domain_classification_loss

        # define train_op
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)

        # wrapper to make the optimizer work with TPUs
        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=global_step)

    if params['use_tpu']:
        # Use host_call to log the losses on the CPU
        def host_call_fn(gs, lpl, dcl, ls):
            gs = gs[0]
            with summary.create_file_writer(params['model_dir'], max_queue=params['save_checkpoints_steps']).as_default():
                with summary.always_record_summaries():
                    summary.scalar('label_prediction_loss', lpl[0], step=gs)
                    summary.scalar('domain_classification_loss', dcl[0], step=gs)
                    summary.scalar('loss', ls[0], step=gs)

            return summary.all_summary_ops()

        # host_call's arguments must be at least 1D
        gs_t = tf.reshape(global_step, [1])
        lpl_t = tf.reshape(label_prediction_loss, [1])
        dcl_t = tf.reshape(domain_classification_loss, [1])
        ls_t = tf.reshape(loss, [1])

        host_call = (host_call_fn, [gs_t, lpl_t, dcl_t, ls_t])

        # TPU version of EstimatorSpec
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            host_call=host_call)
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


# ## The input function
# There are two input data sets, `source` is labeled and `target` is unlabeled.

def train_input_fn(params={}):
    # source distribution: labeled data
    source = np.random.rand(100, 5)
    labels = np.random.randint(0, N_CLASSES, 100)

    # target distribution: unlabeled data
    target = np.random.rand(100, 5)

    source_tensor = tf.constant(source, dtype=tf.float32)
    labels_tensor = tf.constant(labels, dtype=tf.int32)
    target_tensor = tf.constant(target, dtype=tf.float32)

    # shuffle source and target separately
    source_labels_dataset = tf.data.Dataset.from_tensor_slices((source_tensor, labels_tensor)).repeat().shuffle(32)
    target_dataset = tf.data.Dataset.from_tensor_slices(target_tensor).repeat().shuffle(32)

    # zip them together to set shapes
    dataset = tf.data.Dataset.zip((source_labels_dataset, target_dataset))

    # TPUEstimator passes params when calling input_fn
    batch_size = params.get('batch_size', 16)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    # TPUs need to know all dimensions when the graph is built
    # Datasets know the batch size only when the graph is run
    def set_shapes_and_format(source_labels, target):
        source, labels = source_labels

        source_shape = source.get_shape().merge_with([batch_size, None])
        labels_shape = labels.get_shape().merge_with([batch_size])
        target_shape = target.get_shape().merge_with([batch_size, None])

        source.set_shape(source_shape)
        labels.set_shape(labels_shape)
        target.set_shape(target_shape)

        # Also format the dataset with a dict for features
        features = {'source': source, 'target': target}

        return features, labels

    dataset = dataset.map(set_shapes_and_format)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


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

    parser.add_argument(
        '--gr-weight',
        default=1.0,
        help='The weight used in the gradient reversal layer.')
    parser.add_argument(
        '--lambda',
        default=1.0,
        help='The trade-off between label_prediction_loss and domain_classification_loss.')

    args, _ = parser.parse_known_args()

    main(args)
