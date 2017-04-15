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

"""Flowers Sample Cloud Runner.
"""
import argparse
import base64
import datetime
import errno
import io
import json
import multiprocessing
import os
import subprocess
import time
import uuid
import apache_beam as beam
from PIL import Image
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import errors

import trainer.preprocess as preprocess_lib

# Model variables
MODEL_NAME = 'flowers'
TRAINER_NAME = 'trainer-0.1.tar.gz'
METADATA_FILE_NAME = 'metadata.json'
EXPORT_SUBDIRECTORY = 'model'
CONFIG_FILE_NAME = 'config.yaml'
MODULE_NAME = 'trainer.task'
SAMPLE_IMAGE = \
  'gs://cloud-ml-data/img/flower_photos/tulips/4520577328_a94c11e806_n.jpg'

# Number of seconds to wait before sending next online prediction after
# an online prediction fails due to model deployment not being complete.
PREDICTION_WAIT_TIME = 30

def process_args():
  """Define arguments and assign default values to the ones that are not set.

  Returns:
    args: The parsed namespace with defaults assigned to the flags.
  """

  parser = argparse.ArgumentParser(
      description='Runs Flowers Sample E2E pipeline.')
  parser.add_argument(
      '--project',
      default=None,
      help='The project to which the job will be submitted.')
  parser.add_argument(
      '--cloud', action='store_true',
      help='Run preprocessing on the cloud.')
  parser.add_argument(
      '--train_input_path',
      default=None,
      help='Input specified as uri to CSV file for the train set')
  parser.add_argument(
      '--eval_input_path',
      default=None,
      help='Input specified as uri to CSV file for the eval set.')
  parser.add_argument(
      '--eval_set_size',
      default=50,
      help='The size of the eval dataset.')
  parser.add_argument(
      '--input_dict',
      default=None,
      help='Input dictionary. Specified as text file uri. '
      'Each line of the file stores one label.')
  parser.add_argument(
      '--deploy_model_name',
      default='flowerse2e',
      help=('If --cloud is used, the model is deployed with this '
            'name. The default is flowerse2e.'))
  parser.add_argument(
      '--dataflow_sdk_path',
      default=None,
      help=('Path to Dataflow SDK location. If None, Pip will '
            'be used to download the latest published version'))
  parser.add_argument(
      '--max_deploy_wait_time',
      default=600,
      help=('Maximum number of seconds to wait after a model is deployed.'))
  parser.add_argument(
      '--deploy_model_version',
      default='v' + uuid.uuid4().hex[:4],
      help=('If --cloud is used, the model is deployed with this '
            'version. The default is four random characters.'))
  parser.add_argument(
      '--preprocessed_train_set',
      default=None,
      help=('If specified, preprocessing steps will be skipped.'
            'The provided preprocessed dataset wil be used in this case.'
            'If specified, preprocessed_eval_set must also be provided.'))
  parser.add_argument(
      '--preprocessed_eval_set',
      default=None,
      help=('If specified, preprocessing steps will be skipped.'
            'The provided preprocessed dataset wil be used in this case.'
            'If specified, preprocessed_train_set must also be provided.'))
  parser.add_argument(
      '--pretrained_model_path',
      default=None,
      help=('If specified, preprocessing and training steps ares skipped.'
            'The pretrained model will be deployed in this case.'))
  parser.add_argument(
      '--sample_image_uri',
      default=SAMPLE_IMAGE,
      help=('URI for a single Jpeg image to be used for online prediction.'))
  parser.add_argument(
      '--gcs_bucket',
      default=None,
      help=('Google Cloud Storage bucket to be used for uploading intermediate '
            'data')),
  parser.add_argument(
      '--output_dir',
      default=None,
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))
  parser.add_argument(
      '--runtime_version',
      default=os.getenv('CLOUDSDK_ML_DEFAULT_RUNTIME_VERSION', '1.0'),
      help=('Tensorflow version for model training and prediction.'))

  args, _ = parser.parse_known_args()

  if args.cloud and not args.project:
    args.project = get_cloud_project()

  return args


class FlowersE2E(object):
  """The end-2-end pipeline for Flowers Sample."""

  def  __init__(self, args=None):
    if not args:
      self.args = process_args()
    else:
      self.args = args

  def preprocess(self):
    """Runs the pre-processing pipeline.

    It tiggers two Dataflow pipelines in parallel for train and eval.
    Returns:
      train_output_prefix: Path prefix for the preprocessed train dataset.
      eval_output_prefix: Path prefix for the preprocessed eval dataset.
    """

    train_dataset_name = 'train'
    eval_dataset_name = 'eval'

    # Prepare the environment to run the Dataflow pipeline for preprocessing.
    if self.args.dataflow_sdk_path:
      dataflow_sdk = self.args.dataflow_sdk_path
      if dataflow_sdk.startswith('gs://'):
        subprocess.check_call(
            ['gsutil', 'cp', self.args.dataflow_sdk_path, '.'])
        dataflow_sdk = self.args.dataflow_sdk_path.split('/')[-1]
    else:
      dataflow_sdk = None

    subprocess.check_call(['python', 'setup.py', 'sdist', '--format=gztar'])

    trainer_uri = os.path.join(self.args.output_dir, TRAINER_NAME)
    subprocess.check_call(
        ['gsutil', '-q', 'cp', os.path.join('dist', TRAINER_NAME), trainer_uri])

    thread_pool = multiprocessing.pool.ThreadPool(2)

    train_output_prefix = os.path.join(self.args.output_dir, 'preprocessed',
                                       train_dataset_name)
    eval_output_prefix = os.path.join(self.args.output_dir, 'preprocessed',
                                      eval_dataset_name)

    train_args = (train_dataset_name, self.args.train_input_path,
                  train_output_prefix, dataflow_sdk, trainer_uri)
    eval_args = (eval_dataset_name, self.args.eval_input_path,
                 eval_output_prefix, dataflow_sdk, trainer_uri)

    # make a pool to run two pipelines in parallel.
    pipeline_pool = [thread_pool.apply_async(self.run_pipeline, train_args),
                     thread_pool.apply_async(self.run_pipeline, eval_args)]
    _ = [res.get() for res in pipeline_pool]
    return train_output_prefix, eval_output_prefix

  def run_pipeline(self, dataset_name, input_csv, output_prefix,
                   dataflow_sdk_location, trainer_uri):
    """Runs a Dataflow pipeline to preprocess the given dataset.

    Args:
      dataset_name: The name of the dataset ('eval' or 'train').
      input_csv: Path to the input CSV file which contains an image-URI with
                 its labels in each line.
      output_prefix:  Output prefix to write results to.
      dataflow_sdk_location: path to Dataflow SDK package.
      trainer_uri: Path to the Flower's trainer package.
    """
    job_name = ('cloud-ml-sample-flowers-' +
                datetime.datetime.now().strftime('%Y%m%d%H%M%S')  +
                '-' + dataset_name)

    options = {
        'staging_location':
            os.path.join(self.args.output_dir, 'tmp', dataset_name, 'staging'),
        'temp_location':
            os.path.join(self.args.output_dir, 'tmp', dataset_name),
        'project':
            self.args.project,
        'job_name': job_name,
        'extra_packages': [trainer_uri],
        'save_main_session':
            True,
    }
    if dataflow_sdk_location:
      options['sdk_location'] = dataflow_sdk_location

    pipeline_name = 'DataflowRunner' if self.args.cloud else 'DirectRunner'

    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    args = argparse.Namespace(**vars(self.args))
    vars(args)['input_path'] = input_csv
    vars(args)['input_dict'] = self.args.input_dict
    vars(args)['output_path'] = output_prefix
    # execute the pipeline
    with beam.Pipeline(pipeline_name, options=opts) as pipeline:
      preprocess_lib.configure_pipeline(pipeline, args)

  def train(self, train_file_path, eval_file_path):
    """Train a model using the eval and train datasets.

    Args:
      train_file_path: Path to the train dataset.
      eval_file_path: Path to the eval dataset.
    """
    trainer_args = [
        '--output_path', self.args.output_dir,
        '--eval_data_paths', eval_file_path,
        '--eval_set_size', str(self.args.eval_set_size),
        '--train_data_paths', train_file_path
    ]

    if self.args.cloud:
      job_name = 'flowers_model' + datetime.datetime.now().strftime(
          '_%y%m%d_%H%M%S')
      command = [
          'gcloud', 'ml-engine', 'jobs', 'submit', 'training', job_name,
          '--stream-logs',
          '--module-name', MODULE_NAME,
          '--staging-bucket', self.args.gcs_bucket,
          '--region', 'us-central1',
          '--project', self.args.project,
          '--package-path', 'trainer',
          '--runtime-version', self.args.runtime_version,
          '--'
      ] + trainer_args
    else:
      command = [
          'gcloud', 'ml-engine', 'local', 'train',
          '--module-name', MODULE_NAME,
          '--package-path', 'trainer',
          '--',
      ] + trainer_args
    subprocess.check_call(command)

  def deploy_model(self, model_path):
    """Deploys the trained model.

    Args:
      model_path: Path to the trained model.
    """

    create_model_cmd = [
        'gcloud', 'ml-engine', 'models', 'create', self.args.deploy_model_name,
        '--regions', 'us-central1',
        '--project', self.args.project,
    ]

    print create_model_cmd
    subprocess.check_call(create_model_cmd)

    submit = [
        'gcloud', 'ml-engine', 'versions', 'create',
        self.args.deploy_model_version,
        '--model', self.args.deploy_model_name,
        '--origin', model_path,
        '--project', self.args.project,
        '--runtime-version', self.args.runtime_version,
    ]
    if not model_path.startswith('gs://'):
      submit.extend(['--staging-bucket', self.args.gcs_bucket])
    print submit
    subprocess.check_call(submit)

    self.adaptive_wait()

    print 'Deployed %s version: %s' % (self.args.deploy_model_name,
                                       self.args.deploy_model_version)

  def adaptive_wait(self):
    """Waits for a model to be fully deployed.

       It keeps sending online prediction requests until a prediction is
       successful or maximum wait time is reached. It sleeps between requests.
    """
    start_time = datetime.datetime.utcnow()
    elapsed_time = 0
    while elapsed_time < self.args.max_deploy_wait_time:
      try:
        self.predict(self.args.sample_image_uri)
        return
      except Exception as e:
        time.sleep(PREDICTION_WAIT_TIME)
        elapsed_time = (datetime.datetime.utcnow() - start_time).total_seconds()
        continue

  def predict(self, image_uri):
    """Sends a predict request for the deployed model for the given image.

    Args:
      image_uri: The input image URI.
    """
    output_json = 'request.json'
    self.make_request_json(image_uri, output_json)
    cmd = [
        'gcloud', 'ml-engine', 'predict',
        '--model', self.args.deploy_model_name,
        '--version', self.args.deploy_model_version,
        '--json-instances', 'request.json',
        '--project', self.args.project
    ]
    subprocess.check_call(cmd)

  def make_request_json(self, uri, output_json):
    """Produces a JSON request suitable to send to CloudML Prediction API.

    Args:
      uri: The input image URI.
      output_json: File handle of the output json where request will be written.
    """
    def _open_file_read_binary(uri):
      try:
        return file_io.FileIO(uri, mode='rb')
      except errors.InvalidArgumentError:
        return file_io.FileIO(uri, mode='r')

    with open(output_json, 'w') as outf:
      with _open_file_read_binary(uri) as f:
        image_bytes = f.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((299, 299), Image.BILINEAR)
        resized_image = io.BytesIO()
        image.save(resized_image, format='JPEG')
        encoded_image = base64.b64encode(resized_image.getvalue())
        row = json.dumps({'key': uri, 'image_bytes': {'b64': encoded_image}})
        outf.write(row)
        outf.write('\n')

  def run(self):
    """Runs the pipeline."""
    model_path = self.args.pretrained_model_path
    if not model_path:
      train_prefix, eval_prefix = (self.args.preprocessed_train_set,
                                   self.args.preprocessed_eval_set)

      if not train_prefix or not eval_prefix:
        train_prefix, eval_prefix = self.preprocess()
      self.train(train_prefix + '*', eval_prefix + '*')
      model_path = os.path.join(self.args.output_dir, EXPORT_SUBDIRECTORY)
    self.deploy_model(model_path)


def get_cloud_project():
  cmd = [
      'gcloud', '-q', 'config', 'list', 'project',
      '--format=value(core.project)'
  ]
  with open(os.devnull, 'w') as dev_null:
    try:
      res = subprocess.check_output(cmd, stderr=dev_null).strip()
      if not res:
        raise Exception('--cloud specified but no Google Cloud Platform '
                        'project found.\n'
                        'Please specify your project name with the --project '
                        'flag or set a default project: '
                        'gcloud config set project YOUR_PROJECT_NAME')
      return res
    except OSError as e:
      if e.errno == errno.ENOENT:
        raise Exception('gcloud is not installed. The Google Cloud SDK is '
                        'necessary to communicate with the Cloud ML service. '
                        'Please install and set up gcloud.')
      raise


def main():
  pipeline = FlowersE2E()
  pipeline.run()

if __name__ == '__main__':
  main()
