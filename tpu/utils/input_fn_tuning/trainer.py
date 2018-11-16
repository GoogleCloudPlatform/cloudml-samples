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


def model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()

    onehot_labels = tf.one_hot(labels, N_CLASSES)

    flattened = tf.reshape(features, shape=(params['batch_size'], -1))
    hidden = tf.layers.dense(flattened, 100, activation=tf.nn.relu)
    logits = tf.layers.dense(hidden, N_CLASSES)

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


# def image_preprocess(image_bytes):
#     image = _decode_and_random_crop(image_bytes, image_size)
#     image = tf.image.random_flip_left_right(image)
#     image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
#     image = tf.image.convert_image_dtype(
#       image, dtype=tf.bfloat16)

#     return image


def dataset_parser(value):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1)
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

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
    """Statically set the batch_size dimension."""
    images.set_shape(images.get_shape().merge_with(
      tf.TensorShape([None, None, None, batch_size])))
    labels.set_shape(labels.get_shape().merge_with(
      tf.TensorShape([batch_size])))

    return images, labels


# The following sample input_fn is based on the input_fn of
# https://github.com/tensorflow/tpu/tree/master/models/official/resnet
def train_input_fn(params, input_fn_params):
    batch_size = params['batch_size']

    file_pattern = 'gs://cloud-tpu-test-datasets/fake_imagenet/train-*'

    filenames_dataset = tf.data.Dataset.list_files(file_pattern)

    filenames_dataset = filenames_dataset.cache()

    filenames_dataset = filenames_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=32))

    def fetch_tfrecords(filenames):
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=input_fn_params['tfrecord_dataset_buffer_size'],
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

    # Convert TFRecord into (features, labels)
    dataset = tfrecord_dataset.apply(
        tf.contrib.data.map_and_batch(
            dataset_parser,
            batch_size=batch_size,
            num_parallel_calls=input_fn_params['map_and_batch_num_parallel_calls'], 
            drop_remainder=True))

    dataset = dataset.map(
        lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
        num_parallel_calls=input_fn_params['transpose_num_parallel_calls'])

    dataset = dataset.map(partial(set_shapes, batch_size))

    # We could also use tf.contrib.data.AUTOTUNE.
    dataset = dataset.prefetch(buffer_size=input_fn_params['prefetch_buffer_size'])

    return dataset


def main(args):
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
        export_to_tpu=False)

    estimator.train(train_input_fn, max_steps=args.max_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tpu-template')
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000)
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=16)
    parser.add_argument(
        '--save-checkpoints-steps',
        type=int,
        default=100)
    parser.add_argument(
        '--tpu',
        default=None)

    args, _ = parser.parse_known_args()

    # main(args)

    from collections import defaultdict
    input_fn = partial(train_input_fn, input_fn_params=defaultdict(lambda: 4))
    ds = input_fn({'batch_size': 2})

    nb = ds.make_one_shot_iterator().get_next()

    sess = tf.Session()
    fl = sess.run(nb)
