# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import os
import six
import subprocess
import hypertune
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers

MODEL_FILE_NAME = 'chainer.model'

class Net(chainer.Chain):
  def __init__(self):
    super(Net, self).__init__()
    with self.init_scope():
      self.conv1 = L.Convolution2D(1, 10, ksize=5)
      self.conv2 = L.Convolution2D(10, 20, ksize=5)
      self.fc1 = L.Linear(None, 50)
      self.fc2 = L.Linear(None, 10)
      
  def forward(self, x):
    x = F.relu(F.max_pooling_2d(self.conv1(x), 2))
    x = F.relu(F.max_pooling_2d(F.dropout(self.conv2(x)), 2))
    x = F.reshape(F.flatten(x), (-1, 320))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return x

class HpReport(chainer.training.Extension):
  """Trainer extension for hyper parameter tuning with CMLE.
  
  Args:
     log_report (str or LogReport): Log report to accumulate the
        observations. This is either the name of a LogReport extensions
        registered to the trainer, or a LogReport instance to use
        internally.
     global_step: key to epoch
     hyperparameter_metric_tag: user-defined
     metric_value: key to metric
  """
  def __init__(self,
               log_report='LogReport',
               hp_global_step='epoch',
               hp_metric_val='validation/main/loss',
               hp_metric_tag='loss'):
    self._log_report = log_report
    self._log_len = 0  # number of observations already done
  
    self._hp_global_step = hp_global_step
    self._hp_metric_val = hp_metric_val
    self._hp_metric_tag = hp_metric_tag

  def __call__(self, trainer):
    log_report = self._log_report
    if isinstance(log_report, str):
      log_report = trainer.get_extension(log_report)
    elif isinstance(log_report, log_report_module.LogReport):
      log_report(trainer)  # update the log report
    else:
      raise TypeError('log report has a wrong type %s' %
                      type(log_report))

    log = log_report.log
    log_len = self._log_len
    hpt = hypertune.HyperTune()

    while len(log) > log_len:
      target_log = log[log_len]
      hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag=self._hp_metric_tag,
        metric_value=target_log[self._hp_metric_val],
        global_step=target_log[self._hp_global_step])
      log_len += 1
    self.log_len = log_len

def get_args():
  """Argument parser.
  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser(description='Chainer MNIST Example')
  parser.add_argument(
      '--batch-size',
      type=int,
      default=100,
      metavar='N',
      help='input batch size for training (default: 100)')
  parser.add_argument(
      '--test-batch-size',
      type=int,
      default=1000,
      metavar='N',
      help='input batch size for testing (default: 1000)')
  parser.add_argument(
      '--epochs',
      type=int,
      default=10,
      metavar='N',
      help='number of epochs to train (default: 10)')
  parser.add_argument(
      '--lr',
      type=float,
      default=0.01,
      metavar='LR',
      help='learning rate (default: 0.01)')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.5,
      metavar='M',
      help='SGD momentum (default: 0.5)')
  parser.add_argument(
      '--model-dir',
      default=None,
      help='The directory to store the model')
  parser.add_argument(
      '--gpu',
      type=int,
      default=-1,
      help='GPU ID (negative value indicates CPU)')
  parser.add_argument(
      '--resume',
      action='store_true',
      help='Resume training')

  args = parser.parse_args()
  return args

def main():
  # Training settings
  args = get_args()

  # Set up a neural network to train
  model = L.Classifier(Net())

  if args.gpu >= 0:
    # Make a specified GPU current
    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu() # Copy the model to the GPU

  # Setup an optimizer
  optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=args.momentum)
  optimizer.setup(model)

  # Load the MNIST dataset
  train, test = chainer.datasets.get_mnist(ndim=3)
  train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
  test_iter = chainer.iterators.SerialIterator(test, args.test_batch_size,
                                               repeat=False, shuffle=False)

  # Set up a trainer
  updater = training.updaters.StandardUpdater(
      train_iter, optimizer, device=args.gpu)
  trainer = training.Trainer(updater, (args.epochs, 'epoch'))

  # Evaluate the model with the test dataset for each epoch
  trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

  # Write a log of evaluation statistics for each epoch
  trainer.extend(extensions.LogReport())

  # Print selected entries of the log to stdout
  trainer.extend(extensions.PrintReport(
      ['epoch', 'main/loss', 'validation/main/loss',
       'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

  # Send selected entries of the log to CMLE HP tuning system
  trainer.extend(
    HpReport(hp_metric_val='validation/main/loss', hp_metric_tag='my_loss'))

  if args.resume:
    # Resume from a snapshot
    tmp_model_file = os.path.join('/tmp', MODEL_FILE_NAME)
    if not os.path.exists(tmp_model_file):
      subprocess.check_call([
        'gsutil', 'cp', os.path.join(args.model_dir, MODEL_FILE_NAME),
        tmp_model_file])
    if os.path.exists(tmp_model_file):
      chainer.serializers.load_npz(tmp_model_file, trainer)
  
  trainer.run()

  if args.model_dir:
    tmp_model_file = os.path.join('/tmp', MODEL_FILE_NAME)
    serializers.save_npz(tmp_model_file, model)
    subprocess.check_call([
        'gsutil', 'cp', tmp_model_file,
        os.path.join(args.model_dir, MODEL_FILE_NAME)])

if __name__ == '__main__':
  main()
