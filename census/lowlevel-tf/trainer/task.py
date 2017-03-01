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
    self._last_checkpoint = None
    self._checkpoints_since_eval = 0
    self._graph = graph
    with graph.as_default():
      value_dict, update_dict = tf.contrib.metrics.aggregate_metric_map(
          metric_dict)

      self._summary_op = tf.summary.merge([
          tf.summary.scalar(name, value_op)
          for name, value_op in value_dict.iteritems()
      ])

      self._final_ops_dict = value_dict
      self._eval_ops = update_dict.values()

    self._file_writer = tf.summary.FileWriter(checkpoint_dir, graph=graph)

  def after_run(self, run_context, run_values):
    latest = tf.train.latest_checkpoint(self._checkpoint_dir)
    if not latest == self._last_checkpoint:
      self._checkpoints_since_eval += 1

    if not self._checkpoints_since_eval > self._eval_every:
      return
    else:
      self._checkpoints_since_eval = 0

    with self._graph.as_default():
      with tf.Session() as session:
        session.run([tf.local_variables_initializer(), tf.tables_initializer()])
        saver = tf.train.Saver()
        saver.restore(session, latest)


        coord = tf.train.Coordinator(clean_stop_exception_types=(
          tf.errors.CancelledError,))
        tf.train.start_queue_runners(coord=coord, sess=session)
        with coord.stop_on_exception():
          step = 0
          while (self._eval_steps is None
                 or step < self._eval_steps) and not coord.should_stop():
            session.run(self._eval_ops)
            step += 1
            if step % 100 == 0:
              tf.logging.info("On Evaluation Step: {}".format(step))

        gs = session.run(tf.contrib.framework.get_or_create_global_step())
        summaries = session.run(self._summary_op)
        self._file_writer.add_summary(summaries, global_step=gs)
        tf.logging.info(session.run(self._final_ops_dict))


def run(target,
        is_chief,
        max_steps,
        output_dir,
        train_data_path,
        eval_data_path,
        train_batch_size=40,
        eval_batch_size=40,
        eval_num_epochs=1,
        eval_every=100,
        hidden_units=[100, 70, 50, 20],
        eval_steps=None,
        eval_interval_secs=1,
        learning_rate=0.1,
        num_epochs=None):

  if is_chief:
    evaluation_graph = tf.Graph()
    with evaluation_graph.as_default():
      features, labels = model.input_fn(
          eval_data_path,
          num_epochs=None,
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
        output_dir,
        metric_dict,
        evaluation_graph,
        eval_steps=eval_steps,
    )]
  else:
    hooks = []

  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter()):
      features, labels = model.input_fn(
          train_data_path,
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
                                           checkpoint_dir=output_dir,
                                           hooks=hooks,
                                           save_checkpoint_secs=20,
                                           save_summaries_steps=50) as session:
      coord = tf.train.Coordinator(clean_stop_exception_types=(
          tf.errors.CancelledError,))
      tf.train.start_queue_runners(coord=coord, sess=session)
      step = global_step_tensor.eval(session=session)
      with coord.stop_on_exception():
        while (max_steps is None or step < max_steps) and not coord.should_stop():
          step, _ = session.run([global_step_tensor, train_op])



def dispatch(*args, **kwargs):
  """Parse TF_CONFIG to cluster_spec, job_name and task_index."""

  tf_config = os.environ.get('TF_CONFIG')

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
  parser.add_argument('--train_data_path',
                      required=True,
                      type=str,
                      help='Training file location', nargs='+')
  parser.add_argument('--eval_data_path',
                      required=True,
                      type=str,
                      help='Evaluation file location', nargs='+')
  parser.add_argument('--output_dir',
                      required=True,
                      type=str,
                      help='GCS or local dir to write checkpoints and export model')
  parser.add_argument('--max_steps',
                      type=int,
                      default=1000,
                      help='Maximum number of training steps to perform')
  parser.add_argument('--train_batch_size',
                      type=int,
                      default=40,
                      help='Batch size for training steps')
  parser.add_argument('--eval_batch_size',
                      type=int,
                      default=40,
                      help='Batch size for evaluation steps')
  parse_args, unknown = parser.parse_known_args()

  dispatch(**parse_args.__dict__)
