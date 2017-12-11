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

"""A Feed forward neural network using TensorFlow Core APIs.
It implements a binary classifier for Census Income Dataset using both single
and distributed node cluster.
"""

import argparse
import json
import os
import threading

import model

import tensorflow as tf

from tensorflow.python.ops import variables
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.saved_model import signature_constants as sig_constants

tf.logging.set_verbosity(tf.logging.INFO)


class EvaluationRunHook(tf.train.SessionRunHook):
  """EvaluationRunHook performs continuous evaluation of the model.

  Args:
    checkpoint_dir (string): Dir to store model checkpoints
    metric_dir (string): Dir to store metrics like accuracy and auroc
    graph (tf.Graph): Evaluation graph
    eval_frequency (int): Frequency of evaluation every n train steps
    eval_steps (int): Evaluation steps to be performed
  """
  def __init__(self,
               checkpoint_dir,
               metric_dict,
               graph,
               eval_frequency,
               eval_steps=None,
               **kwargs):

    self._eval_steps = eval_steps
    self._checkpoint_dir = checkpoint_dir
    self._kwargs = kwargs
    self._eval_every = eval_frequency
    self._latest_checkpoint = None
    self._checkpoints_since_eval = 0
    self._graph = graph

    # With the graph object as default graph
    # See https://www.tensorflow.org/api_docs/python/tf/Graph#as_default
    # Adds ops to the graph object
    with graph.as_default():
      value_dict, update_dict = tf.contrib.metrics.aggregate_metric_map(
          metric_dict)

      # Op that creates a Summary protocol buffer by merging summaries
      self._summary_op = tf.summary.merge([
          tf.summary.scalar(name, value_op)
          for name, value_op in value_dict.iteritems()
      ])

      # Saver class add ops to save and restore
      # variables to and from checkpoint
      self._saver = tf.train.Saver()

      # Creates a global step to contain a counter for
      # the global training step
      self._gs = tf.contrib.framework.get_or_create_global_step()

      self._final_ops_dict = value_dict
      self._eval_ops = update_dict.values()

    # MonitoredTrainingSession runs hooks in background threads
    # and it doesn't wait for the thread from the last session.run()
    # call to terminate to invoke the next hook, hence locks.
    self._eval_lock = threading.Lock()
    self._checkpoint_lock = threading.Lock()
    self._file_writer = tf.summary.FileWriter(
        os.path.join(checkpoint_dir, 'eval'), graph=graph)

  def after_run(self, run_context, run_values):
    # Always check for new checkpoints in case a single evaluation
    # takes longer than checkpoint frequency and _eval_every is >1
    self._update_latest_checkpoint()

    if self._eval_lock.acquire(False):
      try:
        if self._checkpoints_since_eval > self._eval_every:
          self._checkpoints_since_eval = 0
          self._run_eval()
      finally:
        self._eval_lock.release()

  def _update_latest_checkpoint(self):
    """Update the latest checkpoint file created in the output dir."""
    if self._checkpoint_lock.acquire(False):
      try:
        latest = tf.train.latest_checkpoint(self._checkpoint_dir)
        if not latest == self._latest_checkpoint:
          self._checkpoints_since_eval += 1
          self._latest_checkpoint = latest
      finally:
        self._checkpoint_lock.release()

  def end(self, session):
    """Called at then end of session to make sure we always evaluate."""
    self._update_latest_checkpoint()

    with self._eval_lock:
      self._run_eval()

  def _run_eval(self):
    """Run model evaluation and generate summaries."""
    coord = tf.train.Coordinator(clean_stop_exception_types=(
        tf.errors.CancelledError, tf.errors.OutOfRangeError))

    with tf.Session(graph=self._graph) as session:
      # Restores previously saved variables from latest checkpoint
      self._saver.restore(session, self._latest_checkpoint)

      session.run([
          tf.tables_initializer(),
          tf.local_variables_initializer()
      ])
      tf.train.start_queue_runners(coord=coord, sess=session)
      train_step = session.run(self._gs)

      tf.logging.info('Starting Evaluation For Step: {}'.format(train_step))
      with coord.stop_on_exception():
        eval_step = 0
        while not coord.should_stop() and (self._eval_steps is None or
                                           eval_step < self._eval_steps):
          summaries, final_values, _ = session.run(
              [self._summary_op, self._final_ops_dict, self._eval_ops])
          if eval_step % 100 == 0:
            tf.logging.info("On Evaluation Step: {}".format(eval_step))
          eval_step += 1

      # Write the summaries
      self._file_writer.add_summary(summaries, global_step=train_step)
      self._file_writer.flush()
      tf.logging.info(final_values)


def run(target,
        cluster_spec,
        is_chief,
        train_steps,
        eval_steps,
        job_dir,
        train_files,
        eval_files,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        eval_frequency,
        first_layer_size,
        num_layers,
        scale_factor,
        num_epochs,
        export_format):

  """Run the training and evaluation graph.
  Args:
    target (string): Tensorflow server target
    is_chief (bool): Boolean flag to specify a chief server
    train_steps (int): Maximum number of training steps
    eval_steps (int): Number of steps to run evaluation for at each checkpoint.
      if eval_steps is None, evaluation will run for 1 epoch.
    job_dir (string): Output dir for checkpoint and summary
    train_files (string): List of CSV files to read train data
    eval_files (string): List of CSV files to read eval data
    train_batch_size (int): Batch size for training
    eval_batch_size (int): Batch size for evaluation
    learning_rate (float): Learning rate for Gradient Descent
    eval_frequency (int): Run evaluation frequency every n training steps.
      Do not evaluate too frequently otherwise you will
      pay for performance and do not evaluate too in-frequently
      otherwise you will not know how soon to stop training.
      Use default values to start with
    first_layer_size (int): Size of the first DNN layer
    num_layers (int): Number of hidden layers in the DNN
    scale_factor (float): Decay rate for the size of hidden layers
    num_epochs (int): Maximum number of training data epochs on which to train
    export_format (str): One of 'JSON', 'CSV' or 'EXAMPLE'. The input format
      for the outputed saved_model binary.
  """

  # Calculate the number of hidden units
  hidden_units = [
      max(2, int(first_layer_size * scale_factor**i))
      for i in range(num_layers)
  ]

  # If the server is chief which is `master`
  # In between graph replication Chief is one node in
  # the cluster with extra responsibility and by default
  # is worker task zero. We have assigned master as the chief.
  #
  # See https://youtu.be/la_M6bCV91M?t=1203 for details on
  # distributed TensorFlow and motivation about chief.
  if is_chief:
    tf.logging.info("Created DNN hidden units {}".format(hidden_units))
    evaluation_graph = tf.Graph()
    with evaluation_graph.as_default():

      # Features and label tensors
      features, labels = model.input_fn(
          eval_files,
          num_epochs=None if eval_steps else 1,
          batch_size=eval_batch_size,
          shuffle=False
      )
      # Accuracy and AUROC metrics
      # model.model_fn returns the dict when EVAL mode
      metric_dict = model.model_fn(
          model.EVAL,
          features.copy(),
          labels,
          hidden_units=hidden_units,
          learning_rate=learning_rate
      )

    hooks = [EvaluationRunHook(
        job_dir,
        metric_dict,
        evaluation_graph,
        eval_frequency,
        eval_steps=eval_steps,
    )]
  else:
    hooks = []

  # Create a new graph and specify that as default
  with tf.Graph().as_default():
    # Placement of ops on devices using replica device setter
    # which automatically places the parameters on the `ps` server
    # and the `ops` on the workers
    #
    # See:
    # https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter
    with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):

      # Features and label tensors as read using filename queue
      features, labels = model.input_fn(
          train_files,
          num_epochs=num_epochs,
          batch_size=train_batch_size
      )

      # Returns the training graph and global step tensor
      train_op, global_step_tensor = model.model_fn(
          model.TRAIN,
          features.copy(),
          labels,
          hidden_units=hidden_units,
          learning_rate=learning_rate
      )

    # Creates a MonitoredSession for training
    # MonitoredSession is a Session-like object that handles
    # initialization, recovery and hooks
    # https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession
    with tf.train.MonitoredTrainingSession(master=target,
                                           is_chief=is_chief,
                                           checkpoint_dir=job_dir,
                                           hooks=hooks,
                                           save_checkpoint_secs=20,
                                           save_summaries_steps=50) as session:
      # Global step to keep track of global number of steps particularly in
      # distributed setting
      step = global_step_tensor.eval(session=session)

      # Run the training graph which returns the step number as tracked by
      # the global step tensor.
      # When train epochs is reached, session.should_stop() will be true.
      while (train_steps is None or
             step < train_steps) and not session.should_stop():
        step, _ = session.run([global_step_tensor, train_op])

    # Find the filename of the latest saved checkpoint file
    latest_checkpoint = tf.train.latest_checkpoint(job_dir)

    # Only perform this if chief
    if is_chief:
      build_and_run_exports(latest_checkpoint,
                            job_dir,
                            model.SERVING_INPUT_FUNCTIONS[export_format],
                            hidden_units)


def main_op():
  init_local = variables.local_variables_initializer()
  init_tables = lookup_ops.tables_initializer()
  return control_flow_ops.group(init_local, init_tables)


def build_and_run_exports(latest, job_dir, serving_input_fn, hidden_units):
  """Given the latest checkpoint file export the saved model.

  Args:
    latest (string): Latest checkpoint file
    job_dir (string): Location of checkpoints and model files
    name (string): Name of the checkpoint to be exported. Used in building the
      export path.
    hidden_units (list): Number of hidden units
    learning_rate (float): Learning rate for the SGD
  """

  prediction_graph = tf.Graph()
  exporter = tf.saved_model.builder.SavedModelBuilder(
      os.path.join(job_dir, 'export'))
  with prediction_graph.as_default():
    features, inputs_dict = serving_input_fn()
    prediction_dict = model.model_fn(
        model.PREDICT,
        features.copy(),
        None,  # labels
        hidden_units=hidden_units,
        learning_rate=None  # learning_rate unused in prediction mode
    )
    saver = tf.train.Saver()

    inputs_info = {
        name: tf.saved_model.utils.build_tensor_info(tensor)
        for name, tensor in inputs_dict.iteritems()
    }
    output_info = {
        name: tf.saved_model.utils.build_tensor_info(tensor)
        for name, tensor in prediction_dict.iteritems()
    }
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs_info,
        outputs=output_info,
        method_name=sig_constants.PREDICT_METHOD_NAME
    )

  with tf.Session(graph=prediction_graph) as session:
    session.run([tf.local_variables_initializer(), tf.tables_initializer()])
    saver.restore(session, latest)
    exporter.add_meta_graph_and_variables(
        session,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            sig_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        },
        legacy_init_op=main_op()
    )

  exporter.save()


def dispatch(*args, **kwargs):
  """Parse TF_CONFIG to cluster_spec and call run() method
  TF_CONFIG environment variable is available when running using
  gcloud either locally or on cloud. It has all the information required
  to create a ClusterSpec which is important for running distributed code.
  """

  tf_config = os.environ.get('TF_CONFIG')
  # If TF_CONFIG is not available run local
  if not tf_config:
    return run(target='', cluster_spec=None, is_chief=True, *args, **kwargs)

  tf_config_json = json.loads(tf_config)

  cluster = tf_config_json.get('cluster')
  job_name = tf_config_json.get('task', {}).get('type')
  task_index = tf_config_json.get('task', {}).get('index')

  # If cluster information is empty run local
  if job_name is None or task_index is None:
    return run(target='', cluster_spec=None, is_chief=True, *args, **kwargs)

  cluster_spec = tf.train.ClusterSpec(cluster)
  server = tf.train.Server(cluster_spec,
                           job_name=job_name,
                           task_index=task_index)

  # Wait for incoming connections forever
  # Worker ships the graph to the ps server
  # The ps server manages the parameters of the model.
  #
  # See a detailed video on distributed TensorFlow
  # https://www.youtube.com/watch?v=la_M6bCV91M
  if job_name == 'ps':
    server.join()
    return
  elif job_name in ['master', 'worker']:
    return run(server.target, cluster_spec, is_chief=(job_name == 'master'),
               *args, **kwargs)


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
                      help="""\
                      GCS or local dir for checkpoints, exports, and
                      summaries. Use an existing directory to load a
                      trained model, or a new directory to retrain""")
  parser.add_argument('--train-steps',
                      type=int,
                      help='Maximum number of training steps to perform.')
  parser.add_argument('--eval-steps',
                      help="""\
                      Number of steps to run evalution for at each checkpoint.
                      If unspecified, will run for 1 full epoch over training
                      data""",
                      default=None,
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
                      default=50,
                      help='Perform one evaluation per n steps')
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
  parser.add_argument('--num-epochs',
                      type=int,
                      help='Maximum number of epochs on which to train')
  parser.add_argument('--export-format',
                      type=str,
                      choices=[model.JSON, model.CSV, model.EXAMPLE],
                      default=model.JSON,
                      help="""\
                      Desired input format for the exported saved_model
                      binary.""")
  parser.add_argument('--verbosity',
                      choices=[
                          'DEBUG',
                          'ERROR',
                          'FATAL',
                          'INFO',
                          'WARN'
                      ],
                      default='INFO',
                      help='Set logging verbosity')
  parse_args, unknown = parser.parse_known_args()
  # Set python level verbosity
  tf.logging.set_verbosity(parse_args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[parse_args.verbosity] / 10)
  del parse_args.verbosity

  if unknown:
    tf.logging.warn('Unknown arguments: {}'.format(unknown))

  dispatch(**parse_args.__dict__)
