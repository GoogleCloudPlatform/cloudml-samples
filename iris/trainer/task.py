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
import sys
import tempfile

from . import util
import tensorflow as tf

import google.cloud.ml as ml

from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.session_bundle import manifest_pb2

NUM_CLASSES = 3

DIMENSION = 4

# This determines a single column that is used to obtain features
# after parsing TF.EXamples,
FEATURES_KEY = 'measurements'

# The following keys determine columns from the parsed Examples to be included
# in the output.
TARGET_KEY = 'species'

VOCAB_KEY = 'vocab'

# This is used to map to unparsed tf.Examples so we can output them.
EXAMPLES_KEY = 'examples'

UNKNOWN_LABEL = 'UNKNOWN'

SCORES_COLUMN = 'score'
KEY_COLUMN = 'key'
TARGET_COLUMN = 'target'
LABEL_COLUMN = 'label'

METADATA_HEAD_KEY = 'columns'

OUTPUTS_KEY = 'outputs'
INPUTS_KEY = 'inputs'


def prediction_input_fn(metadata):
  """Input function used by the experiment."""

  def input_fn():
    # Generate placeholders for the examples.
    examples = tf.placeholder(
        dtype=tf.string,
        shape=(None,),
        name='input_example')
    parsed = ml.features.FeatureMetadata.parse_features(metadata, examples)
    parsed[EXAMPLES_KEY] = examples
    # Target is not applicable here, so None is returned.
    return parsed, None

  return input_fn


def file_input_fn(metadata, data_paths, batch_size, shuffle):
  def input_fn():
    _, examples = util.read_examples(data_paths, batch_size, shuffle)
    parsed = ml.features.FeatureMetadata.parse_features(metadata, examples)
    target = parsed.pop(TARGET_KEY)
    return parsed, target

  return input_fn


def get_signature_fn(metadata_path):

  def _build_signature(examples, features, predictions):
    """Create a generic signature function with input and output signatures."""
    iris_labels = VocabGetter(metadata_path).get_vocab().keys()
    prediction = tf.argmax(predictions, 1)
    labels = tf.contrib.lookup.index_to_string(prediction,
                                               mapping=iris_labels,
                                               default_value=UNKNOWN_LABEL)

    target = tf.contrib.lookup.index_to_string(tf.squeeze(features[TARGET_KEY]),
                                               mapping=iris_labels,
                                               default_value=UNKNOWN_LABEL)
    outputs = {SCORES_COLUMN: predictions.name,
               KEY_COLUMN: tf.squeeze(features[KEY_COLUMN]).name,
               TARGET_COLUMN: target.name,
               LABEL_COLUMN: labels.name}

    inputs = {EXAMPLES_KEY: examples.name}

    tf.add_to_collection(OUTPUTS_KEY, json.dumps(outputs))
    tf.add_to_collection(INPUTS_KEY, json.dumps(inputs))

    input_signature = manifest_pb2.Signature()
    output_signature = manifest_pb2.Signature()

    for name, tensor_name in outputs.iteritems():
      output_signature.generic_signature.map[name].tensor_name = tensor_name

    for name, tensor_name in inputs.iteritems():
      input_signature.generic_signature.map[name].tensor_name = tensor_name

    # Return None for default classification signature.
    return None, {INPUTS_KEY: input_signature,
                  OUTPUTS_KEY: output_signature}
  return _build_signature


def make_experiment_fn(train_data_paths, eval_data_paths, metadata_path,
                       max_steps, layer1_size, layer2_size, learning_rate,
                       epsilon, batch_size, eval_batch_size):

  def experiment_fn(output_dir):
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
    metadata = ml.features.FeatureMetadata.get_metadata(
        metadata_path)

    # Specify that all features have real-valued data
    feature_columns = [tf.contrib.layers.real_valued_column(
        FEATURES_KEY, dimension=DIMENSION)]

    train_dir = os.path.join(output_dir, 'train')
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[layer1_size, layer2_size],
        n_classes=NUM_CLASSES,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            learning_rate, epsilon=epsilon))

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
            additional_assets=[metadata_path],
            input_fn=prediction_input_fn(metadata),
            input_feature_key=EXAMPLES_KEY,
            signature_fn=get_signature_fn(metadata_path))
    ]

    train_input_fn = file_input_fn(
        metadata,
        train_data_paths,
        batch_size,
        shuffle=True)

    eval_input_fn = file_input_fn(
        metadata,
        eval_data_paths,
        eval_batch_size,
        shuffle=False)
    streaming_accuracy = metrics_lib.streaming_accuracy
    return tf.contrib.learn.Experiment(
        estimator=classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=max_steps,
        train_monitors=train_monitors,
        min_eval_frequency=1000,
        eval_metrics={
            ('accuracy', 'classes'): streaming_accuracy,
            ('training/hptuning/metric', 'classes'): streaming_accuracy
        })
  return experiment_fn


def parse_arguments(argv):
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
  return parser.parse_args(args=argv[1:])


class VocabGetter(object):

  def __init__(self, metadata_path):
    self.metadata_path = metadata_path
    self.vocab_dic = {}

  def get_vocab(self):
    # Returns a dictionary of Iris labels to arbitrary integer identifiers.
    if not self.vocab_dic:
      yaml_data = ml.features.FeatureMetadata.load_from(self.metadata_path)
      self.vocab_dic = yaml_data[METADATA_HEAD_KEY][TARGET_KEY][VOCAB_KEY]
    return self.vocab_dic


def main(argv=None):
  """Runs a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task') or {'type': 'master', 'index': 0}

  trial = task_data.get('trial')
  if trial is not None:
    output_dir = os.path.join(args.output_path, trial)
  else:
    output_dir = args.output_path

  learn_runner.run(
      experiment_fn=make_experiment_fn(
          train_data_paths=args.train_data_paths,
          eval_data_paths=args.eval_data_paths,
          metadata_path=args.metadata_path,
          max_steps=args.max_steps,
          layer1_size=args.layer1_size,
          layer2_size=args.layer2_size,
          learning_rate=args.learning_rate,
          epsilon=args.epsilon,
          batch_size=args.batch_size,
          eval_batch_size=args.eval_batch_size),
      output_dir=output_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
