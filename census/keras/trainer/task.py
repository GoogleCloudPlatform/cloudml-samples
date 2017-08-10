# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""This code implements a Feed forward neural network using Keras API."""

import argparse
import glob
import json
import os

import keras
from keras import callbacks
from keras.models import load_model
import model

INPUT_SIZE = 55
CLASS_SIZE = 2

# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
CHUNK_SIZE = 5000
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
CENSUS_MODEL = 'census.hdf5'

class ContinuousEval(keras.callbacks.Callback):
  """Continuous eval callback to evaluate the checkpoint once
     every so many epochs.
  """

  def __init__(self,
               eval_frequency,
               eval_files,
               learning_rate,
               job_dir,
               steps=1000):
    self.eval_files = eval_files
    self.eval_frequency = eval_frequency
    self.learning_rate = learning_rate
    self.job_dir = job_dir
    self.steps = steps

  def on_epoch_begin(self, epoch, logs={}):
    if epoch > 0 and epoch % self.eval_frequency == 0:
      checkpoints = glob.glob(os.path.join(self.job_dir, 'checkpoint.*'))
      checkpoints.sort()
      census_model = load_model(checkpoints[-1])
      census_model = model.compile_model(census_model, self.learning_rate)
      loss, acc = census_model.evaluate_generator(
          model.generator_input(self.eval_files, chunk_size=CHUNK_SIZE),
          steps=self.steps)
      print '\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
          epoch, loss, acc, census_model.metrics_names)


def dispatch(train_files,
             eval_files,
             job_dir,
             train_steps,
             eval_steps,
             train_batch_size,
             eval_batch_size,
             learning_rate,
             eval_frequency,
             first_layer_size,
             num_layers,
             scale_factor,
             eval_num_epochs,
             num_epochs,
             checkpoint_epochs):
  census_model = model.model_fn(INPUT_SIZE, CLASS_SIZE)

  try:
    os.makedirs(job_dir)
  except:
    pass

  # Model checkpoint callback
  checkpoint = callbacks.ModelCheckpoint(
      os.path.join(job_dir, FILE_PATH),
      monitor='val_loss',
      verbose=1,
      period=checkpoint_epochs,
      mode='max')

  # Continuous eval callback
  evaluation = ContinuousEval(eval_frequency,
                              eval_files,
                              learning_rate,
                              job_dir)

  # Tensorboard logs callback
  tblog = callbacks.TensorBoard(
      log_dir=os.path.join(job_dir, 'logs'),
      histogram_freq=0,
      write_graph=True,
      embeddings_freq=0)

  #TODO: This needs to be fixed in h5py so that writes to GCS are possible
  # Don't attempt to create checkpoints on Cloud ML Engine for now because
  # h5py doesn't come with native GCS write capability
  if job_dir.startswith('gs://'):
    callbacks=[evaluation, tblog]
  else:
    callbacks=[checkpoint, evaluation, tblog]

  census_model.fit_generator(
      model.generator_input(train_files, chunk_size=CHUNK_SIZE),
      steps_per_epoch=train_steps,
      epochs=num_epochs,
      callbacks=callbacks)

  census_model.save(os.path.join(job_dir, CENSUS_MODEL))

  # Convert the Keras model to TensorFlow SavedModel
  model.to_savedmodel(census_model, os.path.join(job_dir, 'export'))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--train-files',
                      required=True,
                      type=str,
                      help='Training files local or GCS', nargs='+')
  parser.add_argument('--eval-files',
                      required=True,
                      type=str,
                      help='Evaluation files local or GCS', nargs='+')
  parser.add_argument('--job-dir',
                      required=True,
                      type=str,
                      help='GCS or local dir to write checkpoints and export model')
  parser.add_argument('--train-steps',
                      type=int,
                      default=100,
                      help="""\
                       Maximum number of training steps to perform
                       Training steps are in the units of training-batch-size.
                       So if train-steps is 500 and train-batch-size if 100 then
                       at most 500 * 100 training instances will be used to train.
                      """)
  parser.add_argument('--eval-steps',
                      help='Number of steps to run evalution for at each checkpoint',
                      default=100,
                      type=int)
  parser.add_argument('--train-batch-size',
                      type=int,
                      default=40,
                      help='Batch size for training steps')
  parser.add_argument('--eval-batch-size',
                      type=int,
                      default=40,
                      help='Batch size for evaluation steps')
  parser.add_argument('--learning-rate',
                      type=float,
                      default=0.003,
                      help='Learning rate for SGD')
  parser.add_argument('--eval-frequency',
                      default=10,
                      help='Perform one evaluation per n epochs')
  parser.add_argument('--first-layer-size',
                     type=int,
                     default=256,
                     help='Number of nodes in the first layer of DNN')
  parser.add_argument('--num-layers',
                     type=int,
                     default=2,
                     help='Number of layers in DNN')
  parser.add_argument('--scale-factor',
                     type=float,
                     default=0.25,
                     help="""\
                      Rate of decay size of layer for Deep Neural Net.
                      max(2, int(first_layer_size * scale_factor**i)) \
                      """)
  parser.add_argument('--eval-num-epochs',
                     type=int,
                     default=1,
                     help='Number of epochs during evaluation')
  parser.add_argument('--num-epochs',
                      type=int,
                      default=20,
                      help='Maximum number of epochs on which to train')
  parser.add_argument('--checkpoint-epochs',
                      type=int,
                      default=5,
                      help='Checkpoint per n training epochs')
  parse_args, unknown = parser.parse_known_args()

  dispatch(**parse_args.__dict__)
