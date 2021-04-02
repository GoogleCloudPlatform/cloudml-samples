# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sample for Criteo dataset can be run as a wide or deep model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import math
import os
import sys

import tensorflow as tf

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow.contrib.learn.python.learn import learn_runner

DATASETS = ['kaggle', 'large']
KAGGLE, LARGE = DATASETS
MODEL_TYPES = ['linear', 'deep']
LINEAR, DEEP = MODEL_TYPES

CROSSES = 'crosses'
NUM_EXAMPLES = 'num_examples'
L2_REGULARIZATION = 'l2_regularization'

KEY_FEATURE_COLUMN = 'example_id'
TARGET_FEATURE_COLUMN = 'clicked'

CROSS_HASH_BUCKET_SIZE = int(1e6)

MODEL_DIR = 'model'

FORMAT_CATEGORICAL_FEATURE_ID = 'categorical-feature-{}_id'
FORMAT_INT_FEATURE = 'int-feature-{}'

#
# Pipeline config for the two datasets
# The data in the CROSSES is 1-based indexing
#
PIPELINE_CONFIG = {
    KAGGLE: {
        NUM_EXAMPLES:
            45 * 1e6,
        L2_REGULARIZATION:
            60,
        CROSSES: [(27, 31), (33, 37), (27, 29), (4, 6), (19, 36), (19, 22),
                  (19, 33), (6, 9), (10, 5), (19, 35, 36), (30, 36), (30, 11),
                  (20, 30), (19, 22, 28), (27, 31, 39), (1, 8), (11, 5),
                  (11, 7), (25, 2), (26, 27, 31), (38, 5), (19, 22, 11),
                  (37, 5), (24, 11), (13, 4), (19, 8), (27, 31, 33),
                  (17, 19, 36), (31, 3), (26, 5), (30, 12), (27, 31, 2),
                  (11, 9), (15, 34), (19, 26, 36), (27, 36), (30, 5), (23, 37),
                  (13, 3), (31, 6), (26, 8), (30, 33), (27, 36, 37), (1, 6),
                  (17, 30), (20, 23), (27, 31, 35), (26, 1), (26, 27, 36)]
    },
    LARGE: {
        NUM_EXAMPLES:
            4 * 1e9,
        L2_REGULARIZATION:
            500,
        CROSSES: [(19, 12), (10, 12), (10, 11), (32, 12), (30, 1), (36, 39),
                  (13, 3), (26, 32), (15, 23), (10, 9), (20, 25), (16, 26, 32),
                  (11, 12), (30, 10), (15, 38), (10, 6), (39, 8), (39, 10),
                  (19, 28, 12), (15, 37), (26, 7), (11, 5), (14, 39, 8),
                  (11, 2), (12, 4), (28, 1), (26, 32, 11), (26, 10, 7),
                  (22, 30), (15, 24, 38), (20, 10, 12), (32, 9), (15, 8),
                  (32, 4), (26, 3), (29, 30), (22, 30, 39), (22, 30, 36, 39),
                  (22, 26), (20, 11), (4, 9), (26, 12), (12, 13), (32, 6),
                  (39, 11), (15, 26, 32)]
    }
}


def create_parser():
  """Initialize command line parser using arparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset',
      help='Criteo dataset to run training on',
      choices=DATASETS,
      required=True)
  parser.add_argument(
      '--model_type',
      help='Model type to train on',
      choices=MODEL_TYPES,
      default=LINEAR)
  parser.add_argument(
      '--train_data_paths', type=str, action='append', required=True)
  parser.add_argument(
      '--eval_data_paths', type=str, action='append', required=True)
  parser.add_argument(
      '--eval_only_summary_filename', type=str,
      help=('If given, model will not be trained but only evaluated using '
            'the data given in --eval_data_paths, the summary will be '
            'written as json to the given filename. Parameters for models etc. '
            'should be the same as those for training.'))
  parser.add_argument('--output_path', type=str, required=True)
  # The following three parameters are required for tf.Transform.
  parser.add_argument('--raw_metadata_path', type=str, required=True)
  parser.add_argument('--transformed_metadata_path', type=str, required=True)
  parser.add_argument('--transform_savedmodel', type=str, required=True)
  parser.add_argument(
      '--hidden_units',
      nargs='*',
      help='List of hidden units per layer. All layers are fully connected. Ex.'
      '`64 32` means first layer has 64 nodes and second one has 32.',
      default=[512],
      type=int)
  parser.add_argument(
      '--batch_size',
      help='Number of input records used per batch',
      default=30000,
      type=int)
  parser.add_argument(
      '--eval_batch_size',
      help='Number of eval records used per batch',
      default=5000,
      type=int)
  parser.add_argument(
      '--train_steps', help='Number of training steps to perform.', type=int)
  parser.add_argument(
      '--eval_steps',
      help='Number of evaluation steps to perform.',
      type=int,
      default=100)
  parser.add_argument(
      '--train_set_size',
      help='Number of samples on the train dataset.',
      type=int)
  parser.add_argument('--l2_regularization', help='L2 Regularization', type=int)
  parser.add_argument(
      '--num_epochs', help='Number of epochs', default=5, type=int)
  parser.add_argument(
      '--ignore_crosses',
      action='store_true',
      default=False,
      help='Whether to ignore crosses (linear model only).')
  return parser


def feature_columns(config, model_type, vocab_sizes, use_crosses):
  """Return the feature columns with their names and types."""
  result = []
  boundaries = [1.5**j - 0.51 for j in range(40)]

  # TODO(b/35300113): Reduce the range and other duplication between this and
  # preprocessing.

  # TODO(b/35300113): Can iterate over metadata so that we don't need to
  # re-define the schema here?

  for index in range(1, 14):
    column = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column(
            FORMAT_INT_FEATURE.format(index),
            dtype=tf.int64),
        boundaries)
    result.append(column)

  if model_type == LINEAR:
    for index in range(14, 40):
      column_name = FORMAT_CATEGORICAL_FEATURE_ID.format(index)
      vocab_size = vocab_sizes[column_name]
      column = tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name, vocab_size, combiner='sum')
      result.append(column)
    if use_crosses:
      for cross in config[CROSSES]:
        column = tf.contrib.layers.crossed_column(
            [result[index - 1] for index in cross],
            hash_bucket_size=CROSS_HASH_BUCKET_SIZE,
            hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY,
            combiner='sum')
        result.append(column)
  elif model_type == DEEP:
    for index in range(14, 40):
      column_name = FORMAT_CATEGORICAL_FEATURE_ID.format(index)
      vocab_size = vocab_sizes[column_name]
      column = tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name, vocab_size, combiner='sum')
      embedding_size = int(math.floor(6 * vocab_size**0.25))
      embedding = tf.contrib.layers.embedding_column(column,
                                                     embedding_size,
                                                     combiner='mean')
      result.append(embedding)

  return result


def gzip_reader_fn():
  return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def get_transformed_reader_input_fn(transformed_metadata,
                                    transformed_data_paths,
                                    batch_size,
                                    mode):
  """Wrap the get input features function to provide the runtime arguments."""
  return input_fn_maker.build_training_input_fn(
      metadata=transformed_metadata,
      file_pattern=(
          transformed_data_paths[0] if len(transformed_data_paths) == 1
          else transformed_data_paths),
      training_batch_size=batch_size,
      label_keys=[TARGET_FEATURE_COLUMN],
      reader=gzip_reader_fn,
      key_feature_name=KEY_FEATURE_COLUMN,
      reader_num_threads=4,
      queue_capacity=batch_size * 2,
      randomize_input=(mode != tf.contrib.learn.ModeKeys.EVAL),
      num_epochs=(1 if mode == tf.contrib.learn.ModeKeys.EVAL else None))


def get_vocab_sizes():
  """Read vocabulary sizes from the metadata."""
  # TODO(b/35300113) This method will change as we move to tf-transform and use
  # the new schema and statistics protos. For now return a large-ish constant
  # (exact vocabulary size not needed, since we are doing "mod" in tf.Learn).
  # Note that the current workaround might come with a quality sacrifice that
  # should hopefully be lifted soon.
  return {FORMAT_CATEGORICAL_FEATURE_ID.format(index): int(10*1000)
          for index in range(14, 40)}


def get_experiment_fn(args):
  """Wrap the get experiment function to provide the runtime arguments."""
  vocab_sizes = get_vocab_sizes()
  use_crosses = not args.ignore_crosses

  def get_experiment(output_dir):
    """Function that creates an experiment http://goo.gl/HcKHlT.

    Args:
      output_dir: The directory where the training output should be written.
    Returns:
      A `tf.contrib.learn.Experiment`.
    """

    config = PIPELINE_CONFIG.get(args.dataset)
    columns = feature_columns(config, args.model_type, vocab_sizes, use_crosses)

    runconfig = tf.contrib.learn.RunConfig()
    cluster = runconfig.cluster_spec
    num_table_shards = max(1, runconfig.num_ps_replicas * 3)
    num_partitions = max(1, 1 + cluster.num_tasks('worker') if cluster and
                         'worker' in cluster.jobs else 0)

    if args.model_type == LINEAR:
      l2_regularization = args.l2_regularization or config[L2_REGULARIZATION]
      estimator = tf.contrib.learn.LinearClassifier(
          model_dir=output_dir,
          feature_columns=columns,
          optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
              example_id_column=KEY_FEATURE_COLUMN,
              symmetric_l2_regularization=l2_regularization,
              num_loss_partitions=num_partitions,  # workers
              num_table_shards=num_table_shards))  # ps
    elif args.model_type == DEEP:
      estimator = tf.contrib.learn.DNNClassifier(
          hidden_units=args.hidden_units,
          feature_columns=columns,
          model_dir=output_dir)

    l2_regularization = args.l2_regularization or config[L2_REGULARIZATION]

    transformed_metadata = metadata_io.read_metadata(
        args.transformed_metadata_path)
    raw_metadata = metadata_io.read_metadata(args.raw_metadata_path)
    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(
            raw_metadata,
            args.transform_savedmodel,
            raw_label_keys=[TARGET_FEATURE_COLUMN]))
    export_strategy = (
        tf.contrib.learn.utils.make_export_strategy(
            serving_input_fn, exports_to_keep=5,
            default_output_alternative_key=None))

    train_input_fn = get_transformed_reader_input_fn(
        transformed_metadata, args.train_data_paths, args.batch_size,
        tf.contrib.learn.ModeKeys.TRAIN)

    eval_input_fn = get_transformed_reader_input_fn(
        transformed_metadata, args.eval_data_paths, args.batch_size,
        tf.contrib.learn.ModeKeys.EVAL)

    train_set_size = args.train_set_size or config[NUM_EXAMPLES]

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_steps=(args.train_steps or
                     args.num_epochs * train_set_size // args.batch_size),
        eval_steps=args.eval_steps,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        export_strategies=export_strategy,
        min_eval_frequency=500)

  # Return a function to create an Experiment.
  return get_experiment


def main(argv=None):
  """Run a Tensorflow model on the Criteo dataset."""
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task') or {'type': 'master', 'index': 0}
  argv = sys.argv if argv is None else argv
  args = create_parser().parse_args(args=argv[1:])

  trial = task_data.get('trial')
  if trial is not None:
    output_dir = os.path.join(args.output_path, trial)
  else:
    output_dir = args.output_path

  # Do only evaluation if instructed so, or call Experiment's run.
  if args.eval_only_summary_filename:
    experiment = get_experiment_fn(args)(output_dir)
    # Note that evaluation here will appear as 'one_pass' in tensorboard.
    results = experiment.evaluate(delay_secs=0)
    # Converts numpy types to native types for json dumps.
    json_out = json.dumps(
        {key: value.tolist() for key, value in results.iteritems()})
    with tf.Session():
      tf.write_file(args.eval_only_summary_filename, json_out).run()
  else:
    learn_runner.run(experiment_fn=get_experiment_fn(args),
                     output_dir=output_dir)


if __name__ == '__main__':
  main()
