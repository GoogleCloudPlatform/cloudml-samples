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

"""Iris Classification Sample Cloud Runner.
"""
import argparse
import datetime
import os
import subprocess
import uuid

import apache_beam as beam
import tensorflow as tf
import trainer.model as iris

import google.cloud.ml as ml
import google.cloud.ml.dataflow.io.tfrecordio as tfrecordio
import google.cloud.ml.io as io


# Model variables
MODEL_NAME = 'iris'
TRAINER_NAME = 'trainer-1.0.tar.gz'


def _default_project():
  get_project = ['gcloud', 'config', 'list', 'project',
                 '--format=value(core.project)']

  with open(os.devnull, 'w') as dev_null:
    return subprocess.check_output(get_project, stderr=dev_null).strip()


parser = argparse.ArgumentParser(
    description='Runs Training on the Iris model data.')
parser.add_argument('--project_id',
                    help='The project to which the job will be submitted.')
parser.add_argument('--cloud', action='store_true',
                    help='Run preprocessing on the cloud.')
parser.add_argument('--metadata_path',
                    help='The path to the metadata file from preprocessing.')
parser.add_argument('--training_data',
                    default='gs://cloud-ml-data/iris/data_train.csv',
                    help='Data to analyze and encode as training features.')
parser.add_argument('--eval_data',
                    default='gs://cloud-ml-data/iris/data_eval.csv',
                    help='Data to encode as evaluation features.')
parser.add_argument('--predict_data',
                    default='gs://cloud-ml-data/iris/data_predict.csv',
                    help='Data to encode as prediction features.')
parser.add_argument('--output_dir', default=None,
                    help=('Google Cloud Storage or Local directory in which '
                          'to place outputs.'))
parser.add_argument('--deploy_model_name', default='iris',
                    help=('If --cloud is used, the model is deployed with this '
                          'name. The default is iris.'))
parser.add_argument('--deploy_model_version',
                    default='v' + uuid.uuid4().hex[:4],
                    help=('If --cloud is used, the model is deployed with this '
                          'version. The default is four random characters.'))
args, passthrough_args = parser.parse_known_args()

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

TRAINER_URI = os.path.join(args.output_dir, TRAINER_NAME)
MODULE_NAME = 'trainer.task'
EXPORT_SUBDIRECTORY = 'model'


def preprocess(pipeline):
  feature_set = iris.IrisFeatures()

  training_data = beam.io.TextFileSource(
      args.training_data, strip_trailing_newlines=True,
      coder=io.CsvCoder.from_feature_set(feature_set, feature_set.csv_columns))

  eval_data = beam.io.TextFileSource(
      args.eval_data, strip_trailing_newlines=True,
      coder=io.CsvCoder.from_feature_set(feature_set, feature_set.csv_columns))

  predict_data = beam.io.TextFileSource(
      args.predict_data, strip_trailing_newlines=True,
      coder=io.CsvCoder.from_feature_set(feature_set, feature_set.csv_columns,
                                         has_target_columns=False))

  train = pipeline | beam.Read('ReadTrainingData', training_data)
  evaluate = pipeline | beam.Read('ReadEvalData', eval_data)
  predict = pipeline | beam.Read('ReadPredictData', predict_data)

  (metadata, train_features, eval_features, predict_features) = (
      (train, evaluate, predict)
      | 'Preprocess'
      >> ml.Preprocess(feature_set, input_format='csv',
                       format_metadata={'headers': feature_set.csv_columns}))

  # Writes metadata.yaml (text file), features_train, features_eval, and
  # features_eval (TFRecord files)
  (metadata | 'SaveMetadata'
            >> io.SaveMetadata(os.path.join(args.output_dir, 'metadata.yaml')))

  # We turn off sharding of the feature files because the dataset is very small.
  (train_features | 'SaveTrain'
                  >> io.SaveFeatures(
                      os.path.join(args.output_dir, 'features_train')))
  (eval_features | 'SaveEval'
                 >> io.SaveFeatures(
                     os.path.join(args.output_dir, 'features_eval')))
  (predict_features | 'SavePredict'
                    >> io.SaveFeatures(
                        os.path.join(args.output_dir, 'features_predict')))

  return metadata, train_features, eval_features, predict_features


def get_train_parameters(metadata):
  job_args = []
  return {
      'package_uris': [TRAINER_URI],
      'python_module': MODULE_NAME,
      'export_subdir': EXPORT_SUBDIRECTORY,
      'metadata': metadata,
      'label': 'Train',
      'region': 'us-central1',
      'scale_tier': 'STANDARD_1',
      'job_args': job_args
  }


def train(pipeline, train_features=None, eval_features=None, metadata=None):
  if not train_features:
    train_features = (
        pipeline
        | 'ReadTrain'
        >> io.LoadFeatures(os.path.join(args.output_dir, 'features_train*')))
  if not eval_features:
    eval_features = (
        pipeline
        | 'ReadEval'
        >> io.LoadFeatures(os.path.join(args.output_dir, 'features_eval*')))

  trained_model, results = ((train_features, eval_features)
                            | ml.Train(**get_train_parameters(metadata)))

  trained_model | 'SaveModel' >> io.SaveModel(os.path.join(args.output_dir,
                                                           'saved_model'))
  results | io.SaveTrainingJobResult(os.path.join(args.output_dir,
                                                  'train_results'))

  return trained_model, results


def evaluate(pipeline, trained_model=None, eval_features=None):
  if not eval_features:
    eval_features = (
        pipeline
        | 'ReadEval'
        >> io.LoadFeatures(os.path.join(args.output_dir, 'features_eval*')))
  if not trained_model:
    trained_model = (pipeline
                     | 'LoadModel'
                     >> io.LoadModel(os.path.join(args.output_dir,
                                                  'saved_model')))

  # Run our evaluation data through a Batch Evaluation, then pull out just
  # the expected and predicted target values.
  evaluations = (eval_features
                 | 'Evaluate' >> ml.Evaluate(trained_model)
                 | beam.Map('CreateEvaluations', make_evaluation_dict))

  coder = io.CsvCoder(['key', 'target', 'predicted', 'score'],
                      ['target', 'predicted', 'score'])
  write_text_file(evaluations, 'WriteEvaluation', 'model_evaluations', coder)
  return evaluations

def make_evaluation_dict((example, prediction)):
  # When running inside of Dataflow, we don't have our global scope,
  # so import tf here so that we can access it.
  import numpy
  import tensorflow as tf

  tf_example = tf.train.Example()
  tf_example.ParseFromString(example.values()[0])
  feature_map = tf_example.features.feature
  scores = prediction['score']
  prediction = numpy.argmax(scores)
  return {
      'key': feature_map['key'].bytes_list.value[0],
      'target': feature_map['species'].int64_list.value[0],
      'predicted': prediction,
      'score': scores[prediction]
  }

def deploy_model(pipeline, model_name, version_name, trained_model=None):
  if not trained_model:
    trained_model = (pipeline
                     | 'LoadModel'
                     >> io.LoadModel(os.path.join(args.output_dir,
                                                  'saved_model')))

  return trained_model | ml.DeployVersion(model_name, version_name)


def model_analysis(pipeline, evaluation_data=None, metadata=None):
  if not metadata:
    metadata = pipeline | io.LoadMetadata(
        os.path.join(args.output_dir, "metadata.yaml"))
  if not evaluation_data:
    coder = io.CsvCoder(['key', 'target', 'predicted', 'score'],
                        ['target', 'predicted', 'score'])
    evaluation_data = read_text_file(pipeline, 'ReadEvaluation',
                                     'model_evaluations', coder=coder)
  confusion_matrix, precision_recall, logloss = (
        evaluation_data | 'AnalyzeModel' >> ml.AnalyzeModel(metadata))

  confusion_matrix | io.SaveConfusionMatrixCsv(
      os.path.join(args.output_dir, 'analyzer_cm.csv'))
  precision_recall | io.SavePrecisionRecallCsv(
      os.path.join(args.output_dir, 'analyzer_pr.csv'))
  write_text_file(logloss, 'Write Log Loss', 'analyzer_logloss.csv')
  return confusion_matrix, precision_recall, logloss


def get_pipeline_name():
  if args.cloud:
    return 'BlockingDataflowPipelineRunner'
  else:
    return 'DirectPipelineRunner'


def dataflow():
  """Run Preprocessing, Training, Eval, and Prediction as a single Dataflow."""
  print 'Building',TRAINER_NAME,'package.'
  subprocess.check_call(['python', 'setup.py', 'sdist', '--format=gztar'])
  subprocess.check_call(['gsutil', '-q', 'cp',
                         os.path.join('dist', TRAINER_NAME),
                         TRAINER_URI])
  opts = None
  if args.cloud:
    options = {
        'staging_location': os.path.join(args.output_dir, 'tmp', 'staging'),
        'temp_location': os.path.join(args.output_dir, 'tmp'),
        'job_name': ('cloud-ml-sample-iris' + '-'
                     + datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
        'project': args.project_id,
        # Dataflow needs a copy of the version of the cloud ml sdk that
        # is being used.
        'extra_packages': [ml.sdk_location, TRAINER_URI],
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
  else:
    # For local runs, the trainer must be installed as a module.
    subprocess.check_call(['pip', 'install', '--upgrade', '--force-reinstall',
                           '--user', os.path.join('dist', TRAINER_NAME)])

  p = beam.Pipeline(get_pipeline_name(), options=opts)

  # Every function below writes its ouput to a file. The inputs to these
  # functions are also optional; if they are missing, the input values are read
  # from a file. Therefore if running this script multiple times, some steps can
  # be removed to prevent recomputing values.
  metadata, train_features, eval_features, predict_features = preprocess(p)

  trained_model, results = train(p, train_features, eval_features, metadata)

  evaluations = evaluate(p, trained_model, eval_features)

  confusion_matrix, precision_recall, logloss = (
      model_analysis(p, evaluations, metadata))

  if args.cloud:
    deployed = deploy_model(p, args.deploy_model_name,
                            args.deploy_model_version, trained_model)
    # Use our deployed model to run a batch prediction.
    output_uri = os.path.join(args.output_dir, 'batch_prediction_results')
    deployed | "Batch Predict" >> ml.Predict([args.predict_data], output_uri,
                                             region='us-central1',
                                             data_format='TEXT')

    print 'Deploying %s version: %s' % (args.deploy_model_name,
                                        args.deploy_model_version)

  p.run()

  if args.cloud:
    print 'Deployed %s version: %s' % (args.deploy_model_name,
                                        args.deploy_model_version)


def write_text_file(pcollection, label, output_name,
                    coder=beam.coders.ToStringCoder()):
  return pcollection | label >> beam.Write(beam.io.TextFileSink(
      os.path.join(args.output_dir, output_name),
      shard_name_template='',
      coder=coder))


def read_text_file(pipeline, label, input_name,
                   coder=beam.coders.StrUtf8Coder()):
  return pipeline | label >> beam.Read(beam.io.TextFileSource(
      os.path.join(args.output_dir, input_name),
      strip_trailing_newlines=True,
      coder=coder))

def main():
  dataflow()


if __name__ == '__main__':
  main()
