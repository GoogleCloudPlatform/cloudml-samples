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


def model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()

    # there are 10 different labels in the fake data
    embedding_dim = 7
    embedding_table = tf.get_variable('embedding_table', shape=(10, embedding_dim), dtype=tf.float32)

    embeddings = tf.nn.embedding_lookup(embedding_table, features)

    # lstm model
    batch_size = params['batch_size']
    sequence_length = params['sequence_length']

    cell = tf.nn.rnn_cell.BasicLSTMCell(7)
    outputs, final_state = tf.nn.dynamic_rnn(cell, embeddings, dtype=tf.float32)

    # flatten the batch and sequence dimensions
    flattened = tf.reshape(outputs, (batch_size*sequence_length, embedding_dim))
    flattened_logits = tf.layers.dense(flattened, 10)

    logits = tf.reshape(flattened_logits, (batch_size, sequence_length, 10))

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
            train_op=train_op
        )
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op
        )


def train_input_fn(params={}):
    # make some fake data of labels
    data_length = 100
    x = np.random.randint(0, 10, data_length)
    y = np.random.randint(0, 10, data_length)

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

        # TPUs need to know all dimensions when the graph is built 
        x_sequence.set_shape((sequence_length,))
        y_sequence.set_shape((sequence_length,))

        return (x_sequence, y_sequence)

    dataset = dataset.map(get_sequences)

    # TPUEstimator passes params when calling input_fn
    batch_size = params.get('batch_size', 16)
    dataset = dataset.batch(batch_size)

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
            num_shards=8 # using Cloud TPU v2-8
        )

        # use the TPU version of RunConfig
        config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=args.model_dir,
            tpu_config=tpu_config,
            save_checkpoints_steps=100,
            save_summary_steps=100
        )

        # TPUEstimator
        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            config=config,
            params=params,
            train_batch_size=args.batch_size,
            eval_batch_size=32, # FIXME
            export_to_tpu=False
        )
    else:
        config = tf.estimator.RunConfig(model_dir=args.model_dir)

        estimator = tf.estimator.Estimator(
            model_fn,
            config=config,
            params=params
        )

    estimator.train(train_input_fn, max_steps=args.max_steps)

    return estimator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tpu-template'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=5
    )
    parser.add_argument(
        '-batch-size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--use-tpu',
        action='store_true'
    )
    parser.add_argument(
        '--tpu',
        default=None
    )

    args, _ = parser.parse_known_args()

    estimator = main(args)
