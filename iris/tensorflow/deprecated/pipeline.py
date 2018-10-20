#/!/usr/bin/env python
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# This sample has been deprecated.
################################################################################


"""Iris Classification Sample Cloud Runner.
"""
import argparse
import datetime
import os
import subprocess
import uuid
import sys

import apache_beam as beam
import trainer.model as iris
import trainer.task as task

import tensorflow as tf

import google.cloud.ml as ml
import google.cloud.ml.io as io

# Model variables
MODEL_NAME = 'iris'
TRAINER_NAME = 'trainer-1.0.tar.gz'
METADATA_FILE_NAME = 'metadata.json'
MODULE_NAME = 'trainer.task'
EXPORT_SUBDIRECTORY = 'model'

def _default_project():
  get_project = ['gcloud', 'config', 'list', 'project',
                 '--format=value(core.project)']

  with open(os.devnull, 'w') as dev_null:
    return subprocess.check_output(get_project, stderr=dev_null).strip()

def parse_arguments(argv):
  parser = argparse.ArgumentParser(
      description='Runs Training on the Iris model data.')
  parser.add_argument(
      '--project_id', help='The project to which the job will be submitted.')
  parser.add_argument(
      '--cloud', action='store_true', help='Run preprocessing on the cloud.')
  parser.add_argument(
      '--training_data',
      default='gs://cloud-ml-data/iris/data_train.csv',
      help='Data to analyze and encode as training features.')
  parser.add_argument(
      '--eval_data',
      default='gs://cloud-ml-data/iris/data_eval.csv',
      help='Data to encode as evaluation features.')
  parser.add_argument(
      '--predict_data',
      default='gs://cloud-ml-data/iris/data_predict.csv',
      help='Data to encode as prediction features.')
  parser.add_argument(
      '--output_dir',
      default=None,
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))
  parser.add_argument(
      '--deploy_model_name',
      default='iris',
      help=('If --cloud is used, the model is deployed with this '
            'name. The default is iris.'))
  parser.add_argument(
      '--deploy_model_version',
      default='v' + uuid.uuid4().hex[:4],
      help=('If --cloud is used, the model is deployed with this '
            'version. The default is four random characters.'))
  parser.add_argument(
      '--sdk_location',
      default=None,
      help=('Specify the location of the Dataflow SDK. If not specified the '
            'SDK will do the right thing.'))
  parser.add_argument(
      '--endpoint',
      default=None,
      help=('HTTPS endpoint to run training against. If not specified the '
            'SDK will do the right thing.'))
  args, passthrough_args = parser.parse_known_args(args=argv[1:])

  args.trainer_job_args = passthrough_args

  if not args.project_id:
    args.project_id = _default_project()

  if not args.output_dir:
    if args.cloud:
      args.output_dir = os.path.join('gs://' + args.project_id + '-ml',
                                     MODEL_NAME)
    else:
      path = 'output'
      if not os.path.isdir(path):
        os.makedirs(path)
      args.output_dir = path

  args.trainer_uri = os.path.join(args.output_dir, TRAINER_NAME)

  return args


def preprocess(pipeline, training_data, eval_data, predict_data, output_dir):
  """Read in input files, runs ml.Preprocess, and writes preprocessed output.

  Args:
    pipeline: beam pipeline
    training_data, eval_data, predict_data: file paths to input csv files.
    output_dir: file path to where to write all the output files.

  Returns:
    metadata and preprocessed features as pcollections.
  """
  feature_set = iris.IrisFeatures()

  coder_with_target = io.CsvCoder.from_feature_set(feature_set,
                                                   feature_set.csv_columns)
  coder_without_target = io.CsvCoder.from_feature_set(feature_set,
                                                      feature_set.csv_columns,
                                                      has_target_columns=False)

  train = (
      pipeline
      | 'ReadTrainingData'
      >> beam.io.textio.ReadFromText(training_data, coder=coder_with_target))
  evaluate = (
      pipeline
      | 'ReadEvalData'
      >> beam.io.textio.ReadFromText(eval_data, coder=coder_with_target))
  predict = (
      pipeline
      | 'ReadPredictData'
      >> beam.io.textio.ReadFromText(predict_data, coder=coder_without_target))

  # TODO(b/32726166) Update input_format and format_metadata to read from these
  # values directly from the coder.
  (metadata, train_features, eval_features, predict_features) = (
      (train, evaluate, predict)
      | 'Preprocess' >> ml.Preprocess(
          feature_set,
          input_format='csv',
          format_metadata={
              'headers': feature_set.csv_columns
          }))

  # Writes metadata.json - specified through METADATA_FILENAME- (text file),
  # features_train, features_eval, and features_eval (TFRecord files)
  (metadata | 'SaveMetadata'
   >> io.SaveMetadata(os.path.join(output_dir, METADATA_FILE_NAME)))

  # We turn off sharding of the feature files because the dataset is very small.
  (train_features | 'SaveTrain'
   >> io.SaveFeatures(os.path.join(output_dir, 'features_train')))
  (eval_features | 'SaveEval'
   >> io.SaveFeatures(os.path.join(output_dir, 'features_eval')))
  (predict_features | 'SavePredict'
   >> io.SaveFeatures(os.path.join(output_dir, 'features_predict')))

  return metadata, train_features, eval_features, predict_features


def get_train_parameters(trainer_uri, endpoint, metadata, trainer_job_args):
  return {
      'package_uris': [trainer_uri, ml.version.installed_sdk_location],
      'python_module': MODULE_NAME,
      'export_subdir': EXPORT_SUBDIRECTORY,
      'cloud_ml_endpoint': endpoint,
      'metadata': metadata,
      'label': 'Train',
      'region': 'us-central1',
      'scale_tier': 'STANDARD_1',
      'job_args': trainer_job_args
  }


def train(pipeline, output_dir, train_args_dict,
          train_features=None, eval_features=None, metadata=None):
  if not train_features:
    train_features = (
        pipeline
        | 'ReadTrain'
        >> io.LoadFeatures(os.path.join(output_dir, 'features_train*')))
  if not eval_features:
    eval_features = (
        pipeline
        | 'ReadEval'
        >> io.LoadFeatures(os.path.join(output_dir, 'features_eval*')))
  if not metadata:
    metadata = (
        pipeline
        | 'ReadMetadata'
        >> io.LoadMetadata(os.path.join(output_dir, METADATA_FILE_NAME)))

  trained_model, results = (
      (train_features, eval_features)
      | 'Train'
      >> ml.Train(**train_args_dict))

  trained_model | 'SaveModel' >> io.SaveModel(
      os.path.join(output_dir, 'saved_model'))
  results | io.SaveTrainingJobResult(
      os.path.join(output_dir, 'train_results'))

  return trained_model, results


def evaluate(pipeline, output_dir, trained_model=None, eval_features=None):
  if not eval_features:
    eval_features = (
        pipeline
        | 'ReadEval'
        >> io.LoadFeatures(os.path.join(output_dir, 'features_eval*')))
  if not trained_model:
    trained_model = (pipeline
                     | 'LoadModel' >>
                     io.LoadModel(os.path.join(output_dir, 'saved_model')))

  # Run our evaluation data through a Batch Evaluation, then pull out just
  # the expected and predicted target values.
  vocab_loader = LazyVocabLoader(os.path.join(output_dir, METADATA_FILE_NAME))

  evaluations = (eval_features
                 | 'Evaluate' >> ml.Evaluate(trained_model)
                 | 'CreateEvaluations' >> beam.Map(
                     make_evaluation_dict, vocab_loader))
  coder = io.CsvCoder(
      column_names=['key', 'target', 'predicted', 'score', 'target_label',
                    'predicted_label', 'all_scores'],
      numeric_column_names=['target', 'predicted', 'score'])
  (evaluations
   | 'WriteEvaluation'
   >> beam.io.textio.WriteToText(os.path.join(output_dir,
                                              'model_evaluations'),
                                 file_name_suffix='.csv',
                                 coder=coder))
  return evaluations


class LazyVocabLoader(object):
  """Lazy load the vocabulary when needed on the worker."""

  def __init__(self, metadata_path):
    self.metadata_path = metadata_path
    self.vocab = {}  # dict of strings to numbers
    self.reverse_vocab = []  # list of strings.

  def get_vocab(self):
    # Returns a dictionary of Iris labels to consecutive integer identifiers.
    if not self.vocab:
      metadata = ml.features.FeatureMetadata.get_metadata(self.metadata_path)
      self.vocab = metadata.columns['species']['vocab']
    return self.vocab

  def get_reverse_vocab(self):
    # Returns a list of consecutive integer identifiers to Iris labels.
    if not self.reverse_vocab:
      vocab = self.get_vocab()
      self.reverse_vocab = [None] * len(vocab)
      for species_name, index_number in vocab.iteritems():
        self.reverse_vocab[index_number] = species_name
    return self.reverse_vocab


def make_evaluation_dict((input_dict, output_dict), vocab_loader):
  """Make summary dict for evaluation.

  Must contain the schema "target, predicted, score[optional]" for use with
  ml.AnalyzeModel.

  Args:
    input_dict: Input to the TF model ({'input_example:0': tf.Example string})
    output_dict: output of the TF model ({'key': ?, 'score': ?, 'label': ?})
    vocab_loader: loads the species vocab.

  Returns:
    A dict suitable for ml.AnalyzeModel that contains other summary data.
  """
  vocab = vocab_loader.get_vocab()
  reverse_vocab = vocab_loader.get_reverse_vocab()

  scores = output_dict[task.SCORES_OUTPUT_COLUMN]
  predicted_label = output_dict[task.LABEL_OUTPUT_COLUMN]

  ex = tf.train.Example()
  ex.ParseFromString(input_dict.values()[0])
  target = ex.features.feature[task.TARGET_FEATURE_COLUMN].int64_list.value[0]

  return {
      'key': output_dict[task.KEY_OUTPUT_COLUMN],
      'target': target,
      'predicted': vocab[predicted_label],
      'score': max(scores),
      'all_scores': scores,
      'target_label': reverse_vocab[target],
      'predicted_label': predicted_label,
  }


def deploy_model(pipeline, output_dir, endpoint, model_name, version_name,
                 trained_model=None):
  if not trained_model:
    trained_model = (pipeline
                     | 'LoadModel' >>
                     io.LoadModel(os.path.join(output_dir, 'saved_model')))

  return trained_model | ml.DeployVersion(model_name, version_name, endpoint)


def model_analysis(pipeline, output_dir, evaluation_data=None, metadata=None):
  if not metadata:
    metadata = (
        pipeline
        | 'LoadMetadataForAnalysis'
        >> io.LoadMetadata(os.path.join(output_dir, METADATA_FILE_NAME)))
  if not evaluation_data:
    coder = io.CsvCoder(
        column_names=['key', 'target', 'predicted', 'score', 'target_label',
                      'predicted_label', 'all_scores'],
        numeric_column_names=['target', 'predicted', 'score'])
    evaluation_data = (
        pipeline
        | 'ReadEvaluation'
        >> beam.io.ReadFromText(os.path.join(output_dir,
                                             'model_evaluations*'),
                                coder=coder))
  confusion_matrix, precision_recall, logloss = (evaluation_data
                                                 | 'AnalyzeModel'
                                                 >> ml.AnalyzeModel(metadata))

  confusion_matrix | io.SaveConfusionMatrixCsv(
      os.path.join(output_dir, 'analyzer_cm.csv'))
  precision_recall | io.SavePrecisionRecallCsv(
      os.path.join(output_dir, 'analyzer_pr.csv'))
  (logloss
   | 'WriteLogLoss'
   >> beam.io.WriteToText(os.path.join(output_dir,
                                       'analyzer_logloss'),
                         file_name_suffix='.csv'))

  return confusion_matrix, precision_recall, logloss


def get_pipeline_name(cloud):
  if cloud:
    return 'DataflowRunner'
  else:
    return  'DirectRunner'

def main(argv=None):
  """Run Preprocessing, Training, Eval, and Prediction as a single Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)

  print 'Building', TRAINER_NAME, 'package.'
  subprocess.check_call(['python', 'setup.py', 'sdist', '--format=gztar'])
  subprocess.check_call(['gsutil', '-q', 'cp',
                         os.path.join('dist', TRAINER_NAME), args.trainer_uri])
  opts = None
  if args.cloud:
    options = {
        'staging_location': os.path.join(args.output_dir, 'tmp', 'staging'),
        'temp_location': os.path.join(args.output_dir, 'tmp', 'staging'),
        'job_name': ('cloud-ml-sample-iris' + '-' +
                     datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
        'project': args.project_id,
        # Dataflow needs a copy of the version of the cloud ml sdk that
        # is being used.
        'extra_packages': [ml.sdk_location, args.trainer_uri],
        'save_main_session': True
    }
    if args.sdk_location:
      options['sdk_location'] = args.sdk_location
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
  else:
    # For local runs, the trainer must be installed as a module.
    subprocess.check_call(['pip', 'install', '--upgrade', '--force-reinstall',
                           '--user', os.path.join('dist', TRAINER_NAME)])

  p = beam.Pipeline(get_pipeline_name(args.cloud), options=opts)

  # Every function below writes its ouput to a file. The inputs to these
  # functions are also optional; if they are missing, the input values are read
  # from a file. Therefore if running this script multiple times, some steps can
  # be removed to prevent recomputing values.
  metadata, train_features, eval_features, predict_features = (
      preprocess(p,
                 args.training_data,
                 args.eval_data,
                 args.predict_data,
                 args.output_dir))

  train_args_dict = get_train_parameters(args.trainer_uri, args.endpoint,
                                         metadata, args.trainer_job_args)
  trained_model, results = train(p, args.output_dir, train_args_dict,
                                 train_features, eval_features, metadata)

  evaluations = evaluate(p, args.output_dir, trained_model, eval_features)

  confusion_matrix, precision_recall, logloss = (
      model_analysis(p, args.output_dir, evaluations, metadata))

  if args.cloud:
    deployed = deploy_model(p, args.output_dir, args.endpoint,
                            args.deploy_model_name,
                            args.deploy_model_version, trained_model)
    # Use our deployed model to run a batch prediction.
    output_uri = os.path.join(args.output_dir, 'batch_prediction_results')
    deployed | 'Batch Predict' >> ml.Predict(
        [args.predict_data],
        output_uri,
        region='us-central1',
        data_format='TEXT',
        cloud_ml_endpoint=args.endpoint)

    print 'Deploying %s version: %s' % (args.deploy_model_name,
                                        args.deploy_model_version)

  p.run().wait_until_finish()

  if args.cloud:
    print 'Deployed %s version: %s' % (args.deploy_model_name,
                                       args.deploy_model_version)


if __name__ == '__main__':
  main()
