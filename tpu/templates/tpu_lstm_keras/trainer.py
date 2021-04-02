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
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM


def build_model():
    inputs = tf.keras.Input(shape=(5, 3))
    encoded = tf.keras.layers.LSTM(10)(inputs)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(encoded)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def train_input_fn():
    batch_size = 16

    # make some fake data
    x = np.random.rand(100, 5, 3)
    y = np.random.rand(100, 1)

    # TPUs currently do not support float64
    x_tensor = tf.constant(x, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.float32)

    # create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

    dataset = dataset.repeat().shuffle(32).batch(batch_size, drop_remainder=True)

    # TPUs need to know all dimensions when the graph is built
    # Datasets know the batch size only when the graph is run
    def set_shapes(features, labels):
        features_shape = features.get_shape().merge_with([batch_size, None, None])
        labels_shape = labels.get_shape().merge_with([batch_size, None])

        features.set_shape(features_shape)
        labels.set_shape(labels_shape)

        return features, labels

    dataset = dataset.map(set_shapes)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def main(args):
    model = build_model()

    if args.use_tpu:
        # distribute over TPU cores
        # Note: This requires TensorFlow 1.11
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu)
        strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu_cluster_resolver)
        model = tf.contrib.tpu.keras_to_tpu_model(
            model, strategy=strategy)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)
    loss_fn = tf.losses.log_loss
    model.compile(optimizer, loss_fn)

    model.fit(train_input_fn, epochs=3, steps_per_epoch=10)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model.save(os.path.join(args.model_dir, 'model.hd5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tpu-template',
        help='Location to write checkpoints and summaries to.  Must be a GCS URI when using Cloud TPU.')
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
