# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Example implementation of code to run on the Cloud ML service.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import tempfile


from . import model as iris
from . import util
import tensorflow as tf

from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn import learn_runner

import google.cloud.ml.features as features

NUM_CLASSES = 3
INPUT_FEATURE_KEY = ''


def file_input_fn(metadata, data_paths, batch_size, shuffle):
  def input_fn():
    # TODO(rhaertel): consider learn.read_batch_examples.
    _, examples = util.read_examples(data_paths, batch_size, shuffle)

    # Get examples and labels from the dataset.
    _, measurements, labels, _ = (
        iris.create_inputs(metadata, input_data=examples))
    return {INPUT_FEATURE_KEY: measurements}, labels

  return input_fn


def prediction_input_fn(metadata):
  def input_fn():
    # Generate placeholders for the examples.
    placeholder, measurements, _, keys = iris.create_inputs(metadata)

    # Mark the inputs and the outputs
    tf.add_to_collection('inputs',
                         json.dumps({'examples': placeholder.name}))
    return {INPUT_FEATURE_KEY: measurements}, None

  return input_fn


def make_signature(examples, unused_features, predictions):
  """Create a classification signature function and add output placeholders."""
  # TODO(b/31436089) Also return the predicted class. This would require the
  # estimator to return a dictionary of predictions and scores that can be added
  # to the outputs.

  # Mark the outputs.
  outputs = {'score': predictions.name}
  tf.add_to_collection('outputs', json.dumps(outputs))

  return tf.contrib.learn.utils.export.classification_signature_fn(
      examples, unused_features, predictions)


class ExperimentFn(object):

  def __init__(self, args):
    self._args = args

  def __call__(self, output_dir):
    """Experiment function used by learn_runner to run training/eval/etc.

    Args:
      output_dir: String path of directory to use for outputs (model
        checkpoints, summaries, etc).

    Returns:
      tf.learn `Experiment`.
    """
    config = tf.contrib.learn.RunConfig()
    # Write checkpoints more often for more granular evals, since the toy data
    # set is so small and simple. Most normal use cases should not set this and
    # just use the default (600).
    config.save_checkpoints_secs = 60

    # Load the metadata.
    metadata = features.FeatureMetadata.get_metadata(self._args.metadata_path)

    # Specify that all features have real-valued data
    feature_columns = [tf.contrib.layers.real_valued_column('', dimension=4)]

    train_dir = os.path.join(output_dir, 'train')
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[self._args.layer1_size, self._args.layer2_size],
        n_classes=NUM_CLASSES,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            self._args.learning_rate, epsilon=self._args.epsilon))

    # In order to export the final model to a predetermined location on GCS,
    # we use a Monitor, specifically ExportLastModelMonitor.
    final_model_dir = os.path.join(output_dir, 'model')

    # GCS lacks atomic directory renaming. Instead, we export models to
    # a temporary location on local disk and then copy that model out
    # to GCS.
    export_dir = tempfile.mkdtemp()
    train_monitors = [
        util.ExportLastModelMonitor(
            export_dir=export_dir,
            dest=final_model_dir,
            additional_assets=[self._args.metadata_path],
            input_fn=prediction_input_fn(metadata),
            input_feature_key=INPUT_FEATURE_KEY,
            signature_fn=make_signature)
    ]

    train_input_fn = file_input_fn(
        metadata,
        self._args.train_data_paths,
        self._args.batch_size,
        shuffle=True)
    eval_input_fn = file_input_fn(
        metadata,
        self._args.eval_data_paths,
        self._args.eval_batch_size,
        shuffle=False)
    streaming_accuracy = metrics_lib.streaming_accuracy
    return tf.contrib.learn.Experiment(
        estimator=classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=self._args.max_steps,
        train_monitors=train_monitors,
        min_eval_frequency=1000,
        eval_metrics={
            ('accuracy', 'classes'): streaming_accuracy,
            ('training/hptuning/metric', 'classes'): streaming_accuracy
        })


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_data_paths', type=str, action='append')
  parser.add_argument('--eval_data_paths', type=str, action='append')
  parser.add_argument('--metadata_path', type=str)
  parser.add_argument('--output_path', type=str)
  parser.add_argument('--max_steps', type=int, default=5000)
  parser.add_argument('--layer1_size', type=int, default=20)
  parser.add_argument('--layer2_size', type=int, default=10)
  parser.add_argument('--learning_rate', type=float, default=0.01)
  parser.add_argument('--epsilon', type=float, default=0.0005)
  parser.add_argument('--batch_size', type=int, default=30)
  parser.add_argument('--eval_batch_size', type=int, default=30)
  return parser.parse_args()


def main():
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task') or {'type': 'master', 'index': 0}

  args = parse_arguments()

  trial = task_data.get('trial')
  if trial is not None:
    output_dir = os.path.join(args.output_path, trial)
  else:
    output_dir = args.output_path

  learn_runner.run(
      experiment_fn=ExperimentFn(args),
      output_dir=output_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
