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

NUM_CLASSES = 10
EMBEDDING_DIM = 7


def model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()

    embedding_table = tf.get_variable('embedding_table', shape=(NUM_CLASSES, EMBEDDING_DIM), dtype=tf.float32)

    embeddings = tf.nn.embedding_lookup(embedding_table, features)

    # lstm model
    batch_size = params['train_batch_size']
    sequence_length = params['sequence_length']

    cell = tf.nn.rnn_cell.BasicLSTMCell(EMBEDDING_DIM)
    outputs, final_state = tf.nn.dynamic_rnn(cell, embeddings, dtype=tf.float32)

    # flatten the batch and sequence dimensions
    flattened = tf.reshape(outputs, (-1, EMBEDDING_DIM))
    flattened_logits = tf.layers.dense(flattened, NUM_CLASSES)

    logits = tf.reshape(flattened_logits, (-1, sequence_length, NUM_CLASSES))

    predictions = tf.multinomial(flattened_logits, num_samples=1)
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

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
    # make some fake data of labels
    data_length = 100
    x = np.random.randint(0, NUM_CLASSES, data_length)
    y = np.random.randint(0, NUM_CLASSES, data_length)

    x_tensor = tf.constant(x, dtype=tf.int32)
    y_tensor = tf.constant(y, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensors((x_tensor, y_tensor))
    dataset = dataset.repeat()

    # TPUs need to know the full shape of tensors
    # so we use a fixed sequence length
    sequence_length = params.get('sequence_length', 5)

    def get_sequences(x_tensor, y_tensor):
        index = tf.random_uniform([1], minval=0, maxval=data_length-sequence_length, dtype=tf.int32)[0]

        x_sequence = x_tensor[index:index+sequence_length]
        y_sequence = y_tensor[index:index+sequence_length]

        return (x_sequence, y_sequence)

    dataset = dataset.map(get_sequences)

    # TPUEstimator passes params when calling input_fn
    batch_size = params.get('train_batch_size', 16)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # TPUs need to know all dimensions when the graph is built
    # Datasets know the batch size only when the graph is run
    def set_shapes(features, labels):
        features_shape = features.get_shape().merge_with([batch_size, sequence_length])
        labels_shape = labels.get_shape().merge_with([batch_size, sequence_length])

        features.set_shape(features_shape)
        labels.set_shape(labels_shape)

        return features, labels

    dataset = dataset.map(set_shapes)
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
        '--sequence-length',
        type=int,
        default=5,
        help='The sequence length for an LSTM model.')
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
        help='The name or GRPC URL of the TPU node.  Leave it as `None` when training on CMLE.')

    args, _ = parser.parse_known_args()

    main(args)
