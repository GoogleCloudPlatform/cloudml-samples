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
from functools import partial
import os
import tensorflow as tf

N_CLASSES = 1000
IMAGE_SIZE = 224

# The input_fn parameters that will be tuned.
input_fn_param_names = [
    'tfrecord_dataset_buffer_size',
    'tfrecord_dataset_num_parallel_reads',
    'parallel_interleave_cycle_length',
    'parallel_interleave_block_length',
    'parallel_interleave_buffer_output_elements',
    'parallel_interleave_prefetch_input_elements',
    'map_and_batch_num_parallel_calls',
    'transpose_num_parallel_calls',
    'prefetch_buffer_size'
]


def model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()

    onehot_labels = tf.one_hot(labels, N_CLASSES)

    # Use bfloat16
    with tf.contrib.tpu.bfloat16_scope():
        flattened = tf.reshape(features, shape=(params['batch_size'], -1))
        hidden = tf.layers.dense(flattened, 100, activation=tf.nn.relu)
        logits = tf.layers.dense(hidden, N_CLASSES)

    logits = tf.cast(logits, tf.float32)

    predictions = tf.nn.softmax(logits)
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)

        # define train_op
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05)
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


def dataset_parser(value):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1)
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    # Preprocess the images.
    image = tf.image.decode_jpeg(image_bytes)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16)

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 1

    return image, label


def set_shapes(batch_size, images, labels):
    # The images are transposed.
    images.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3, batch_size])
    labels.set_shape([batch_size])

    return images, labels


# The following sample input_fn is based on the input_fn of
# https://github.com/tensorflow/tpu/tree/master/models/official/resnet
def _train_input_fn(params, input_fn_params):
    batch_size = params['batch_size']

    file_pattern = 'gs://cloud-tpu-test-datasets/fake_imagenet/train-*'

    filenames_dataset = tf.data.Dataset.list_files(file_pattern)

    filenames_dataset = filenames_dataset.cache()

    filenames_dataset = filenames_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=32))

    def fetch_tfrecords(filenames):
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=1e6*input_fn_params['tfrecord_dataset_buffer_size'],
            num_parallel_reads=input_fn_params['tfrecord_dataset_num_parallel_reads'])
        return dataset

    # Get data from GCS.
    tfrecord_dataset = filenames_dataset.apply(
        tf.contrib.data.parallel_interleave(fetch_tfrecords, 
            cycle_length=input_fn_params['parallel_interleave_cycle_length'], 
            block_length=input_fn_params['parallel_interleave_block_length'], 
            sloppy=True, 
            buffer_output_elements=input_fn_params['parallel_interleave_buffer_output_elements'], 
            prefetch_input_elements=input_fn_params['parallel_interleave_prefetch_input_elements']))

    # Convert TFRecord into (features, labels) tuple.
    dataset = tfrecord_dataset.apply(
        tf.contrib.data.map_and_batch(
            dataset_parser,
            batch_size=batch_size,
            num_parallel_calls=input_fn_params['map_and_batch_num_parallel_calls'], 
            drop_remainder=True))

    # Always transpose the images.
    dataset = dataset.map(
        lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
        num_parallel_calls=input_fn_params['transpose_num_parallel_calls'])

    dataset = dataset.map(partial(set_shapes, batch_size, ))

    # We could also use tf.contrib.data.AUTOTUNE.
    dataset = dataset.prefetch(buffer_size=input_fn_params['prefetch_buffer_size'])

    return dataset


def main(args):
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu)
    tpu_config = tf.contrib.tpu.TPUConfig(
        num_shards=8, # using Cloud TPU v2-8
        iterations_per_loop=args.save_checkpoints_steps)

    config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=args.model_dir,
        tpu_config=tpu_config,
        save_checkpoints_steps=args.save_checkpoints_steps,
        save_summary_steps=100)

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        config=config,
        # params=params,
        train_batch_size=args.train_batch_size,
        export_to_tpu=False)

    # Extract input_fn_params from args.
    input_fn_params = {input_fn_param_name: getattr(args, input_fn_param_name) for input_fn_param_name in input_fn_param_names}

    # Build the input_fn.
    train_input_fn = partial(_train_input_fn, input_fn_params=input_fn_params)

    estimator.train(train_input_fn, max_steps=args.max_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/tpu-template')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000)
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=16)
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=100)
    parser.add_argument(
        '--tpu',
        default=None)

    # Expose input_fn_params to the command line.
    for input_fn_param_name in input_fn_param_names:
        parser.add_argument('--{}'.format(input_fn_param_name), type=int)

    args, _ = parser.parse_known_args()

    main(args)
