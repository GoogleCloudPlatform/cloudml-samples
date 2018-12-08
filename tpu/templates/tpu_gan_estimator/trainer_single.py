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

INPUT_DIM = 5
OUTPUT_DIM = 3

def generator_fn(generator_inputs):
    outputs = tf.layers.dense(generator_inputs, OUTPUT_DIM)
    return outputs


def discriminator_fn(data, generator_inputs):
    outputs = tf.layers.dense(data, 1)
    return outputs


def model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()

    generator_inputs = features
    real_data = labels

    gan_model = tf.contrib.gan.gan_model(generator_fn, discriminator_fn, real_data, generator_inputs)

    predictions = gan_model.generated_data
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        gan_loss = tf.contrib.gan.gan_loss(gan_model, add_summaries=False)
        loss = gan_loss.generator_loss

        # define train_op
        gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)
        dis_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)

        # wrapper to make the optimizer work with TPUs
        if params['use_tpu']:
            gen_optimizer = tf.contrib.tpu.CrossShardOptimizer(gen_optimizer)
            dis_optimizer = tf.contrib.tpu.CrossShardOptimizer(dis_optimizer)

        gan_train_ops = tf.contrib.gan.gan_train_ops(gan_model, gan_loss, gen_optimizer, dis_optimizer)

        while_loop = tf.contrib.tpu.while_loop if params['use_tpu'] else tf.while_loop

        # train the discriminator 100 steps
        inputs = [tf.constant(0), tf.constant(0.0)]
        cond = lambda i, x: tf.less(i, 100)
        def body(i, x):
            return tf.add(i, 1), gan_train_ops.discriminator_train_op

        dis_train_op = while_loop(cond, body, inputs)

        # tf.contrib.gan's train op does not manage global steps in it
        train_op = tf.group(
            dis_train_op,
            gan_train_ops.generator_train_op,
            global_step.assign_add(1))

    if params['use_tpu']:
        # TPU version of EstimatorSpec
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,)
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,)


def train_input_fn(params={}):
    # make some fake noise
    data_size = 100
    noise_tensor = tf.random_normal((data_size, INPUT_DIM))
    real_data_tensor = tf.random_uniform((data_size, OUTPUT_DIM))

    dataset = tf.data.Dataset.from_tensor_slices((noise_tensor, real_data_tensor))
    dataset = dataset.repeat().shuffle(10)

    # TPUEstimator passes params when calling input_fn
    batch_size = params.get('train_batch_size', 16)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # TPUs need to know all dimensions when the graph is built
    # Datasets know the batch size only when the graph is run
    def set_shapes(features, labels):
        features_shape = features.get_shape().merge_with([batch_size, None])
        labels_shape = labels.get_shape().merge_with([batch_size, None])

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
            save_summary_steps=10)

        # TPUEstimator
        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            config=config,
            params=params,
            train_batch_size=args.train_batch_size,
            eval_batch_size=32,
            export_to_tpu=False)
        
    else:
        config = tf.estimator.RunConfig(
            model_dir=args.model_dir,
            save_checkpoints_steps=10,
            save_summary_steps=10)

        estimator = tf.estimator.Estimator(
            model_fn,
            config=config,
            params=params)

    estimator.train(train_input_fn, steps=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tpu-template',
        help='Location to write checkpoints and summaries to.  Must be a GCS URI when using Cloud TPU.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=16,
        help='The training batch size.  The training batch is divided evenly across the TPU cores.')
    parser.add_argument(
        '--save-checkpoints-steps',
        type=int,
        default=10,
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
