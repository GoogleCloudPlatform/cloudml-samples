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
"""Trains MNIST by feeding data using a single worker.

This version is based on:

https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/mnist/fully_connected_feed.py

but adds support for training and prediction on the Google Cloud ML service.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import json
import os.path
import subprocess
import tempfile
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


from tensorflow.core.protobuf import meta_graph_pb2
# Copy of tensorflow.examples.tutorials.mnist.input_data but includes
# support for keys.
from trainer import input_data

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_string('model_dir', 'data', 'Directory to put the model into.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_string('input_path', None,
                    'The GCS path to download the train and eval files from. '
                    'If this is not set, then they will be downloaded from '
                    'a default HTTP address. This path must contain the files '
                    'listed in INPUT_FILES')

INPUT_FILES = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
               't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']


def placeholder_inputs():
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Returns:
    keys_placeholder: Keys placeholder.
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets. We set
  # batch_size to None in order to allow us to use a variable number of
  # instances for online prediction.
  keys_placeholder = tf.placeholder(tf.int64, shape=(None,))
  images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(None,))
  return keys_placeholder, images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  _, images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                    FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST. If input_path is specified, download the data from GCS to
  # the folder expected by read_data_sets.
  data_dir = tempfile.mkdtemp()
  if FLAGS.input_path:
    files = [os.path.join(FLAGS.input_path, file_name)
             for file_name in INPUT_FILES]
    subprocess.check_call(['gsutil', '-m', '-q', 'cp', '-r'] + files +
                          [data_dir])
  data_sets = input_data.read_data_sets(data_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels and mark as input.
    placeholders = placeholder_inputs()
    keys_placeholder, images_placeholder, labels_placeholder = placeholders
    inputs = {'key': keys_placeholder.name, 'image': images_placeholder.name}
    input_signatures = {}
    for key, val in inputs.iteritems():
      predict_input_tensor = meta_graph_pb2.TensorInfo()
      predict_input_tensor.name = val
      for placeholder in placeholders:
        if placeholder.name == val:
          predict_input_tensor.dtype = placeholder.dtype.as_datatype_enum
      input_signatures[key] = predict_input_tensor

    tf.add_to_collection('inputs', json.dumps(inputs))

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # To be able to extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)

    # The prediction will be the index in logits with the highest score.
    # We also use a softmax operation to produce a probability distribution
    # over all possible digits.
    prediction = tf.argmax(logits, 1)
    scores = tf.nn.softmax(logits)

    # Mark the outputs.
    outputs = {'key': keys.name,
               'prediction': prediction.name,
               'scores': scores.name}
    output_signatures = {}
    for key, val in outputs.iteritems():
      predict_output_tensor = meta_graph_pb2.TensorInfo()
      predict_output_tensor.name = val
      for placeholder in [keys, prediction, scores]:
        if placeholder.name == val:
          predict_output_tensor.dtype = placeholder.dtype.as_datatype_enum
      output_signatures[key] = predict_output_tensor

    tf.add_to_collection('outputs', json.dumps(outputs))

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    # Remove this if once Tensorflow 0.12 is standard.
    try:
      summary_op = tf.contrib.deprecated.merge_all_summaries()
    except AttributeError:
      summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing legacy training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    # Remove this if once Tensorflow 0.12 is standard.
    try:
      summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    except AttributeError:
      summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)

    file_io.create_dir(FLAGS.model_dir)

    predict_signature_def = signature_def_utils.build_signature_def(
        input_signatures, output_signatures,
        signature_constants.PREDICT_METHOD_NAME)

    # Create a saver for writing SavedModel training checkpoints.
    build = builder.SavedModelBuilder(
        os.path.join(FLAGS.model_dir, 'saved_model'))
    logging.debug('Saved model path %s', os.path.join(FLAGS.model_dir,
                                                      'saved_model'))
    build.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                predict_signature_def
        },
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
    build.save()


def main(_):
  logging.info('Current tf version: %s', tf.__version__)
  logging.info('Current tf git version: %s', tf.__git_version__)
  run_training()


if __name__ == '__main__':
  tf.app.run()
