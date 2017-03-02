# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""This code implements a Feed forward neural network using TF low level APIs.
   It implements a binary classifier for Census Income Dataset using both single
   and distributed node cluster.
"""

import argparse
import json
import os
import threading

import tensorflow as tf

import model

tf.logging.set_verbosity(tf.logging.INFO)


class EvalRepeatedlyHook(tf.train.SessionRunHook):

  def __init__(self,
               checkpoint_dir,
               metric_dict,
               graph,
               eval_every_n_checkpoints=1,
               eval_steps=None,
               **kwargs):

    self._eval_steps = eval_steps
    self._checkpoint_dir = checkpoint_dir
    self._kwargs = kwargs
    self._eval_every = eval_every_n_checkpoints
    self._latest_checkpoint = None
    self._checkpoints_since_eval = 0
    self._graph = graph

    with graph.as_default():
      value_dict, update_dict = tf.contrib.metrics.aggregate_metric_map(
          metric_dict)

      self._summary_op = tf.summary.merge([
          tf.summary.scalar(name, value_op)
          for name, value_op in value_dict.iteritems()
      ])
      self._saver = tf.train.Saver()
      self._gs = tf.contrib.framework.get_or_create_global_step()
      self._final_ops_dict = value_dict
      self._eval_ops = update_dict.values()

    self._eval_lock = threading.Lock()
    self._checkpoint_lock = threading.Lock()
    self._file_writer = tf.summary.FileWriter(
        os.path.join(checkpoint_dir, 'eval'), graph=graph)

  def after_run(self, run_context, run_values):
    # Always check for new checkpoints in case a single evaluation
    # takes longer than checkpoint frequency and _eval_every is >1
    self._update_latest_checkpoint()
    if self._eval_lock.acquire(False):
      if self._checkpoints_since_eval > self._eval_every:
        self._checkpoints_since_eval = 0
        self._run_eval()

      self._eval_lock.release()

  def _update_latest_checkpoint(self):
    if self._checkpoint_lock.acquire(False):
      latest = tf.train.latest_checkpoint(self._checkpoint_dir)
      if not latest == self._latest_checkpoint:
        self._checkpoints_since_eval += 1
        self._latest_checkpoint = latest
      self._checkpoint_lock.release()

  def end(self, session):
    # Block to ensure we always eval at the end
    self._update_latest_checkpoint()
    self._eval_lock.acquire()
    self._run_eval()
    self._eval_lock.release()

  def _run_eval(self):
    coord = tf.train.Coordinator(clean_stop_exception_types=(
        tf.errors.CancelledError, tf.errors.OutOfRangeError))

    with tf.Session(graph=self._graph) as session:
      self._saver.restore(session, self._latest_checkpoint)

      session.run([
          tf.local_variables_initializer(),
          tf.tables_initializer()
      ])
      tf.train.start_queue_runners(coord=coord, sess=session)
      train_step = session.run(self._gs)

      tf.logging.info('Starting Evaluation For Step: {}'.format(train_step))
      with coord.stop_on_exception():
        eval_step = 0
        while self._eval_steps is None or eval_step < self._eval_steps:
          summaries, final_values, _ = session.run(
              [self._summary_op, self._final_ops_dict, self._eval_ops])
          if eval_step % 100 == 0:
            tf.logging.info("On Evaluation Step: {}".format(eval_step))
          eval_step += 1

      self._file_writer.add_summary(summaries, global_step=train_step)
      tf.logging.info(final_values)

def run(target,
        is_chief,
        max_steps,
        job_dir,
        train_data_paths,
        eval_data_paths,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        first_layer_size,
        num_layers,
        scale_factor,
        eval_num_epochs=1,
        eval_every=100,
        eval_steps=None,
        eval_interval_secs=1,
        num_epochs=None):

  """Run the training and evaluation graph.

  Args:
    target (string): Tensorflow server target
    is_chief (bool): Boolean flag to specify a chief server
    max_steps (int): Maximum training steps
    job_dir (string): Output dir for checkpoint and summary
    train_data_paths (string): List of CSV files to read train data
    eval_data_paths (string): List of CSV files to read eval data
    train_batch_size (int): Batch size for training
    eval_batch_size (int): Batch size for evaluation
    learning_rate (float): Learning rate for Gradient Descent
    first_layer_size (int): Size of the first DNN layer
    num_layers (int): Number of hidden layers in the DNN
    scale_factor (float): Decay rate for the size of hidden layers
    eval_every (int): Run evaluation frequency
    eval_steps (int): Eval steps
    num_epochs (int): Number of epochs
  """

  # Calculate the number of hidden units
  hidden_units=[
      max(2, int(first_layer_size * scale_factor**i))
      for i in range(num_layers)
  ]

  if is_chief:
    tf.logging.info("Created DNN hidden units {}".format(hidden_units))
    evaluation_graph = tf.Graph()
    with evaluation_graph.as_default():
      features, labels = model.input_fn(
          eval_data_paths,
          num_epochs=eval_num_epochs,
          batch_size=eval_batch_size,
          shuffle=False
      )

      metric_dict = model.model_fn(
          model.EVAL,
          features,
          labels,
          hidden_units=hidden_units,
          learning_rate=learning_rate
      )
    hooks = [EvalRepeatedlyHook(
        job_dir,
        metric_dict,
        evaluation_graph,
        eval_steps=eval_steps,
    )]
  else:
    hooks = []

  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter()):
      features, labels = model.input_fn(
          train_data_paths,
          num_epochs=num_epochs,
          batch_size=train_batch_size
      )

      train_op, global_step_tensor = model.model_fn(
          model.TRAIN,
          features,
          labels,
          hidden_units=hidden_units,
          learning_rate=learning_rate
      )


    with tf.train.MonitoredTrainingSession(master=target,
                                           is_chief=is_chief,
                                           checkpoint_dir=job_dir,
                                           hooks=hooks,
                                           save_checkpoint_secs=2,
                                           save_summaries_steps=50) as session:
      coord = tf.train.Coordinator(clean_stop_exception_types=(
          tf.errors.CancelledError,))
      tf.train.start_queue_runners(coord=coord, sess=session)
      step = global_step_tensor.eval(session=session)
      with coord.stop_on_exception():
        while (max_steps is None or step < max_steps) and not coord.should_stop():
          step, _ = session.run([global_step_tensor, train_op])

    latest_checkpoint = tf.train.latest_checkpoint(job_dir)
    for mode in ['CSV', 'TF_RECORD', 'JSON']:
      build_and_run_exports(latest_checkpoint, job_dir, mode, hidden_units, learning_rate)


def build_and_run_exports(latest, job_dir, mode, hidden_units, learning_rate):
  prediction_graph = tf.Graph()
  exporter = tf.saved_model.builder.SavedModelBuilder(
      os.path.join(job_dir, 'exports', mode))
  with prediction_graph.as_default():
    features, placeholder_dict = model.build_serving_inputs(mode)
    prediction_dict = model.model_fn(
        model.PREDICT,
        features,
        None,  # labels
        hidden_units=hidden_units,
        learning_rate=learning_rate
    )
    saver = tf.train.Saver()

    placeholder_info = {
        name: tf.saved_model.utils.build_tensor_info(tensor)
        for name, tensor in placeholder_dict.iteritems()
    }
    output_info = {
        name: tf.saved_model.utils.build_tensor_info(tensor)
        for name, tensor in prediction_dict.iteritems()
    }
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=placeholder_info,
        outputs=output_info,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )


  with tf.Session(graph=prediction_graph) as session:
    session.run([tf.local_variables_initializer(), tf.tables_initializer()])
    saver.restore(session, latest)
    exporter.add_meta_graph_and_variables(
        session,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        },
        main_op=tf.tables_initializer()
    )

  exporter.save()


def dispatch(*args, **kwargs):
  """Parse TF_CONFIG to cluster_spec and call run()."""

  tf_config = os.environ.get('TF_CONFIG')

  # If TF_CONFIG not available run local
  if not tf_config:
    return run('', True, *args, **kwargs)

  tf_config_json = json.loads(tf_config)

  cluster = tf_config_json.get('cluster')
  job_name = tf_config_json.get('task').get('type')
  task_index = tf_config_json.get('task').get('index')

  # If cluster information is empty run local
  if job_name is None or task_index is None:
    return run('', True, *args, **kwargs)

  cluster_spec = tf.train.ClusterSpec(cluster)
  server = tf.train.Server(cluster_spec,
                           job_name=job_name,
                           task_index=task_index)

  if job_name == 'ps':
    server.join()
    return
  elif job_name in ['master', 'worker']:
    return run(server.target, job_name == 'master', *args, **kwargs)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--train-data-paths',
                      required=True,
                      type=str,
                      help='Training file location', nargs='+')
  parser.add_argument('--eval-data-paths',
                      required=True,
                      type=str,
                      help='Evaluation file location', nargs='+')
  parser.add_argument('--job-dir',
                      required=True,
                      type=str,
                      help='GCS or local dir to write checkpoints and export model')
  parser.add_argument('--max-steps',
                      type=int,
                      default=1000,
                      help='Maximum number of training steps to perform')
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
                      default=0.1,
                      help='Learning rate for SGD')
  parser.add_argument('--first-layer-size',
                     type=int,
                     default=100,
                     help='Number of nodes in the first layer of DNN')
  parser.add_argument('--num-layers',
                     type=int,
                     default=4,
                     help='Number of layers in DNN')
  parser.add_argument('--scale-factor',
                     type=float,
                     default=0.7,
                     help='Rate of decay size of layer for DNN')
  parse_args, unknown = parser.parse_known_args()

  dispatch(**parse_args.__dict__)
