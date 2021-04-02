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
from tensorflow.contrib.tensorboard.plugins import projector

tf.logging.set_verbosity(tf.logging.INFO)

PREDICT_BATCH_SIZE = 2000
EMBEDDING_SIZE = 64

# Triplet loss metric learning with TPU based on https://arxiv.org/abs/1503.03832

def model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()
    hidden = tf.layers.dense(features, 100, activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden, EMBEDDING_SIZE)

    # normalize
    embeddings = tf.math.l2_normalize(outputs, axis=1)

    # TPUEstimatorSpec.predictions must be dict of Tensors.
    predictions = {'embeddings': embeddings}

    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, embeddings)

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


def train_input_fn(params={}):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0

    # TPUs currently do not support float64
    x_tensor = tf.constant(x_train, dtype=tf.float32)
    x_tensor = tf.reshape(x_tensor, (-1, 28*28))

    y_tensor = tf.constant(y_train, dtype=tf.int32)

    # create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

    # TPUEstimator passes params when calling input_fn
    batch_size = params.get('batch_size', 256)

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


def predict_input_fn(params={}):
    batch_size = params.get('predict_batch_size', PREDICT_BATCH_SIZE)

    mnist = tf.keras.datasets.mnist
    _, (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0

    x_test = x_test[:batch_size]
    y_test = y_test[:batch_size]

    x_tensor = tf.constant(x_test, dtype=tf.float32)
    x_tensor = tf.reshape(x_tensor, (-1, 28*28))

    y_tensor = tf.constant(y_test, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensors((x_tensor, y_tensor))

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
            # Calling TPUEstimator.predict requires setting predict_bath_size.
            predict_batch_size=PREDICT_BATCH_SIZE,
            eval_batch_size=32,
            export_to_tpu=False)
    else:
        config = tf.estimator.RunConfig(model_dir=args.model_dir)

        estimator = tf.estimator.Estimator(
            model_fn,
            config=config,
            params=params)

    estimator.train(train_input_fn, max_steps=args.max_steps)

    # After training, apply the learned embedding to the test data and visualize with tensorboard Projector.
    embeddings = next(estimator.predict(predict_input_fn, yield_single_examples=False))['embeddings']

    # Put the embeddings into a variable to be visualized.
    embedding_var = tf.Variable(embeddings, name='test_embeddings')

    # Labels do not pass through the estimator.predict call, so we get it separately.
    _, (_, labels) = tf.keras.datasets.mnist.load_data()
    labels = labels[:PREDICT_BATCH_SIZE]

    # Write the metadata file for the projector.
    metadata_path = os.path.join(estimator.model_dir, 'metadata.tsv')
    with tf.gfile.GFile(metadata_path, 'w') as f:
        f.write('index\tlabel\n')
        for i, label in enumerate(labels):
            f.write('{}\t{}\n'.format(i, label))

    # Configure the projector.
    projector_config = projector.ProjectorConfig()
    embedding_config = projector_config.embeddings.add()
    embedding_config.tensor_name = embedding_var.name

    # The metadata_path is relative to the summary_writer's log_dir.
    embedding_config.metadata_path = 'metadata.tsv'

    summary_writer = tf.summary.FileWriter(estimator.model_dir)

    projector.visualize_embeddings(summary_writer, projector_config)

    # Start a session to actually write the embeddings into a new checkpoint.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(estimator.model_dir, 'model.ckpt'), args.max_steps+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tpu-triplet-loss',
        help='Location to write checkpoints and summaries to.  Must be a GCS URI when using Cloud TPU.')
    parser.add_argument(
        '--max-steps',
        type=int,
        default=3000,
        help='The total number of steps to train the model.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=128,
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
