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

EVAL = 'EVAL'
TRAIN = 'TRAIN'


def run(target,
        is_chief,
        max_steps,
        train_data_paths,
        eval_data_paths,
        job_dir,
        eval_every=100,
        eval_steps=10,
        learning_rate=0.1,
        num_epochs=None,
        batch_size=40):
  """Run the training steps and calculate accuracy every 10 steps."""

  training_eval_graph = tf.Graph()
  with training_eval_graph.as_default():
    with tf.device(tf.train.replica_device_setter()):
      mode = tf.placeholder(shape=[], dtype=tf.string)
      eval_features, eval_label = model.input_fn(
          eval_data_paths, shuffle=False, batch_size=batch_size)
      train_features, train_label = model.input_fn(
          train_data_paths, num_epochs=num_epochs, batch_size=batch_size)

      is_train = tf.equal(mode, tf.constant(TRAIN))
      sorted_keys = train_features.keys()
      sorted_keys.sort()
      inputs = dict(zip(
          sorted_keys,
          tf.cond(
              is_train,
              lambda: [train_features[k] for k in sorted_keys],
              lambda: [eval_features[k] for k in sorted_keys]
          )
      ))
      labels = tf.cond(is_train, lambda: train_label, lambda: eval_label)
      train_op, accuracy_op, global_step_tensor, predictions = model.model_fn(
          inputs, labels, learning_rate=learning_rate)

    with tf.train.MonitoredTrainingSession(master=target,
                                           is_chief=is_chief,
                                           checkpoint_dir=job_dir,
                                           save_checkpoint_secs=20,
                                           save_summaries_steps=50) as session:
      coord = tf.train.Coordinator(clean_stop_exception_types=(
          tf.errors.CancelledError,))
      tf.train.start_queue_runners(coord=coord, sess=session)
      step = 0
      last_eval = 0
      with coord.stop_on_exception():
        while step < max_steps and not coord.should_stop():
            if is_chief and step - last_eval > eval_every:
                last_eval = step
                accuracies = [
                    session.run(accuracy_op, feed_dict={mode: EVAL})
                    for _ in range(eval_steps)
                ]
                accuracy = sum(accuracies) / eval_steps
                print("Accuracy at step: {} is {:.2f}%".format(step, 100*accuracy))

            step, _ = session.run(
                [global_step_tensor, train_op],
                feed_dict={mode: TRAIN}
            )


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
  parser.add_argument(
      '--train_data_paths', required=True, type=str,
      help='Training file location', nargs='+')
  parser.add_argument(
      '--eval_data_paths', required=True, type=str,
      help='Evaluation file location', nargs='+')
  parser.add_argument(
      '--job_dir', required=True, type=str,
      help='Location to write checkpoints and export model'
  )
  parser.add_argument('--max_steps', type=int, default=1000,
      help='Maximum number of training steps to perform')
  parse_args, unknown = parser.parse_known_args()

  dispatch(**parse_args.__dict__)
