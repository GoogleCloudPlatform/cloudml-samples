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
"""Implements a DNN, using a custom tf.estimator.Estimator"""

# See https://goo.gl/JZ6hlH to contrast this with DNN combined
# which the "canned" estimator based sample implements.
import multiprocessing
import six

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

# See tutorial on wide and deep
# https://www.tensorflow.org/tutorials/wide_and_deep/

# Define the format of your input data including unused columns
CSV_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'gender',
               'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'income_bracket']
CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                       [0], [0], [0], [''], ['']]
LABEL_COLUMN = 'income_bracket'
LABELS = [' <=50K', ' >50K']

# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
    # Categorical base columns

    # For categorical columns with known values we can provide lists
    # of values ahead of time.
    tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', [' Female', ' Male']),

    tf.feature_column.categorical_column_with_vocabulary_list(
        'race',
        [' Amer-Indian-Eskimo', ' Asian-Pac-Islander',
         ' Black', ' Other', ' White']
    ),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'education',
        [' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
         ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',
         ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th',
         ' 1st-4th', ' Preschool', ' 12th']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status',
        [' Married-civ-spouse', ' Divorced', ' Married-spouse-absent',
         ' Never-married', ' Separated', ' Married-AF-spouse', ' Widowed']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',
        [' Husband', ' Not-in-family', ' Wife', ' Own-child', ' Unmarried',
         ' Other-relative']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass',
        [' Self-emp-not-inc', ' Private', ' State-gov',
         ' Federal-gov', ' Local-gov', ' ?', ' Self-emp-inc',
         ' Without-pay', ' Never-worked']
    ),

    # For columns with a large number of values, or unknown values
    # We can use a hash function to convert to categories.
    tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=100, dtype=tf.string),
    tf.feature_column.categorical_column_with_hash_bucket(
        'native_country', hash_bucket_size=100, dtype=tf.string),

    # Continuous base columns.
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('education_num'),
    tf.feature_column.numeric_column('capital_gain'),
    tf.feature_column.numeric_column('capital_loss'),
    tf.feature_column.numeric_column('hours_per_week'),
]

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - \
    {LABEL_COLUMN}


def generate_model_fn(embedding_size=8,
                      hidden_units=[100, 70, 40, 20],
                      learning_rate=0.1):
  """Generates a model_fn for a feed forward classification network.

  Takes hyperparameters that define the model and returns a model_fn that
  generates a spec from input Tensors.

  Args:
    hidden_units (list): Hidden units of the DNN.
    learning_rate (float): Learning rate for the SGD.
    embedding_size (int): Dimenstionality of embeddings for high dimension
       categorical columns.

  Returns:
    A model_fn.
    See https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
    for details on the signature of the model_fn.
  """
  def _model_fn(mode, features, labels):
    """A model_fn that builds the DNN classification spec

    Args:
      mode (tf.estimator.ModeKeys): One of ModeKeys.(TRAIN|PREDICT|INFER) which
         is used to selectively add operations to the graph.
      features (Mapping[str:Tensor]): Input features for the model.
      labels (Tensor): Label Tensor.

    Returns:
      tf.estimator.EstimatorSpec which defines the model. Will have different
      populated members depending on `mode`. See:
        https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec
      for details.
    """
    (gender, race, education, marital_status, relationship,
     workclass, occupation, native_country, age,
     education_num, capital_gain, capital_loss, hours_per_week) = INPUT_COLUMNS

    transformed_columns = [
        # Use indicator columns for low dimensional vocabularies
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(gender),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.indicator_column(race),

        # Use embedding columns for high dimensional vocabularies
        tf.feature_column.embedding_column(
            native_country, dimension=embedding_size),
        tf.feature_column.embedding_column(
            occupation, dimension=embedding_size),
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
    ]

    inputs = tf.feature_column.input_layer(features, transformed_columns)
    label_values = tf.constant(LABELS)

    # Build the DNN
    curr_layer = inputs

    for layer_size in hidden_units:
      curr_layer = tf.layers.dense(
          curr_layer,
          layer_size,
          activation=tf.nn.relu,
          # This initializer prevents variance from exploding or vanishing when
          # compounded through different sized layers.
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
      )

    # Add the output layer
    logits = tf.layers.dense(
        curr_layer,
        len(LABELS),
        # Do not use ReLU on last layer
        activation=None,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )

    if mode in (Modes.PREDICT, Modes.EVAL):
      probabilities = tf.nn.softmax(logits)
      predicted_indices = tf.argmax(probabilities, 1)

    if mode in (Modes.TRAIN, Modes.EVAL):
      # Convert the string label column to indices
      # Build a lookup table inside the graph
      table = tf.contrib.lookup.index_table_from_tensor(label_values)

      # Use the lookup table to convert string labels to ints
      label_indices = table.lookup(labels)
      # Make labels a vector
      label_indices_vector = tf.squeeze(label_indices)

      # global_step is necessary in eval to correctly load the step
      # of the checkpoint we are evaluating
      global_step = tf.contrib.framework.get_or_create_global_step()
      loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits, labels=label_indices_vector))
      tf.summary.scalar('loss', loss)

    if mode == Modes.PREDICT:
      # Convert predicted_indices back into strings
      predictions = {
          'classes': tf.gather(label_values, predicted_indices),
          'scores': tf.reduce_max(probabilities, axis=1)
      }
      export_outputs = {
          'prediction': tf.estimator.export.PredictOutput(predictions)
      }
      return tf.estimator.EstimatorSpec(
          mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
      # Build training operation.
      train_op = tf.train.FtrlOptimizer(
          learning_rate=learning_rate,
          l1_regularization_strength=3.0,
          l2_regularization_strength=10.0
      ).minimize(loss, global_step=global_step)
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
      # Return accuracy and area under ROC curve metrics
      # See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
      # See https://www.kaggle.com/wiki/AreaUnderCurve
      labels_one_hot = tf.one_hot(
          label_indices_vector,
          depth=label_values.shape[0],
          on_value=True,
          off_value=False,
          dtype=tf.bool
      )
      eval_metric_ops = {
          'accuracy': tf.metrics.accuracy(label_indices, predicted_indices),
          'auroc': tf.metrics.auc(labels_one_hot, probabilities)
      }
      return tf.estimator.EstimatorSpec(
          mode, loss=loss, eval_metric_ops=eval_metric_ops)
  return _model_fn


def csv_serving_input_fn():
  """Build the serving inputs."""
  csv_row = tf.placeholder(
      shape=[None],
      dtype=tf.string
  )
  features = parse_csv(csv_row)
  # Ignore label column
  features.pop(LABEL_COLUMN)
  return tf.estimator.export.ServingInputReceiver(
      features, {'csv_row': csv_row})


def example_serving_input_fn():
  """Build the serving inputs."""
  example_bytestring = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  features = tf.parse_example(
      example_bytestring,
      tf.feature_column.make_parse_example_spec(INPUT_COLUMNS)
  )
  return tf.estimator.export.ServingInputReceiver(
      features, {'example_proto': example_bytestring})


def json_serving_input_fn():
  """Build the serving inputs."""
  inputs = {}
  for feat in INPUT_COLUMNS:
    inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}


def parse_csv(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""
  columns = tf.decode_csv(
      rows_string_tensor, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)

  for key, value in six.iteritems(features):
    features[key] = tf.expand_dims(features[key], -1)
  return features


def generate_input_fn(filenames,
                      num_epochs=None,
                      shuffle=True,
                      skip_header_lines=0,
                      batch_size=200):
  """Generates an input function for training or evaluation.

  This uses the input pipeline based approach using file name queue
  to read data so that entire data is not loaded in memory.

  Args:
      filenames: [str] list of CSV files to read data from.
      num_epochs: int how many times through to read the data.
        If None will loop through data indefinitely
      shuffle: bool, whether or not to randomize the order of data.
        Controls randomization of both file order and line order within
        files.
      skip_header_lines: int set to non-zero in order to skip header lines
        in CSV files.
      batch_size: int First dimension size of the Tensors returned by
        input_fn
  Returns:
      A function () -> (Mapping[str:Tensor], Tensor) which produces the
      feature dictionary, and label Tensor.
  """
  filename_queue = tf.train.string_input_producer(

      filenames, num_epochs=num_epochs, shuffle=shuffle)
  reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

  _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

  # Parse the CSV File
  features = parse_csv(rows)

  # This operation builds up a buffer of parsed tensors, so that parsing
  # input data doesn't block training. If requested it will also shuffle.
  if shuffle:
    features = tf.train.shuffle_batch(
        features,
        batch_size,
        min_after_dequeue=2 * batch_size + 1,
        capacity=batch_size * 10,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
        allow_smaller_final_batch=True
    )
  else:
    features = tf.train.batch(
        features,
        batch_size,
        capacity=batch_size * 10,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
        allow_smaller_final_batch=True
    )

  return features, features.pop(LABEL_COLUMN)
