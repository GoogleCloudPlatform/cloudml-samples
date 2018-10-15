from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import os
import tensorflow as tf

from input_metadata import CATEGORICAL_FEATURE_KEYS_TO_BE_REMOVED
from input_metadata import HASH_STRING_FEATURE_KEYS
from input_metadata import LABEL_KEY
from input_metadata import NUMERIC_FEATURE_KEYS
from input_metadata import NUMERIC_FEATURE_KEYS_TO_BE_REMOVED
from input_metadata import RAW_DATA_METADATA
from input_metadata import STRING_TO_INT_FEATURE_KEYS
from input_metadata import TO_BE_BUCKETIZED_FEATURE

from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import metadata_io


def vocabulary_file_by_name(working_dir, key):
  """Retrieves the path of the vocabulary created by tf transform
       and named after the feature_column

    Args:
      working_dir : Directory pointed by the tf transform pipeline
      key : Name of the feature_column

    Return:
      The path to the vocabulary
  """
  return os.path.join(
      working_dir,
      transform_fn_io.TRANSFORM_FN_DIR,
      'assets',
      key)


def vocabulary_size_by_name(working_dir, key):
  """Retrieves the size of the vocabulary created by tf transform
     and named after the feature_column

  Args:
    working_dir : Directory pointed by the tf transform pipeline
    key : Name of the feature_column

  Return:
    The size of the vocabulary
  """
  vocabulary = vocabulary_file_by_name(working_dir, key)
  with tf.gfile.Open(vocabulary, 'r') as f:
    return sum(1 for _ in f)


# Functions for training
def _make_training_input_fn(tft_working_dir,
                            filebase,
                            num_epochs=None,
                            shuffle=True,
                            batch_size=200,
                            buffer_size=None,
                            prefetch_buffer_size=1):
  """Creates an input function reading from transformed data.

  Args:
    tft_working_dir: Directory to read transformed data and metadata from and to
        write exported model to.
    filebase: Base filename (relative to `tft_working_dir`) of examples.
    num_epochs: int how many times through to read the data.
      If None will loop through data indefinitely
    shuffle: bool, whether or not to randomize the order of data.
      Controls randomization of both file order and line order within
      files.
    batch_size: Batch size
    buffer_size: Buffer size for the shuffle
    prefetch_buffer_size: Number of example to prefetch

  Returns:
    The input function for training or eval.
  """
  if buffer_size is None:
    buffer_size = 2 * batch_size + 1

  # Examples have already been transformed so we only need the feature_columns
  # to parse the single the tf.Record

  transformed_metadata = metadata_io.read_metadata(
      os.path.join(
          tft_working_dir, transform_fn_io.TRANSFORMED_METADATA_DIR))
  transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

  def parser(record):
    """Help function to parse tf.Example"""
    parsed = tf.parse_single_example(record, transformed_feature_spec)
    label = parsed.pop(LABEL_KEY)
    return parsed, label

  def input_fn():
    """Input function for training and eval."""
    files = tf.data.Dataset.list_files(os.path.join(
        tft_working_dir, filebase + '*'))
    dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=4, block_length=16)
    dataset = dataset.map(parser)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(prefetch_buffer_size)
    iterator = dataset.make_one_shot_iterator()
    transformed_features, transformed_labels = iterator.get_next()

    return transformed_features, transformed_labels

  return input_fn


def _make_serving_input_fn(tft_working_dir):
  """Creates an input function from serving.

  Args:
    tft_working_dir: Directory to read transformed data and metadata from and to
        write exported model to.

  Returns:
    The input function for serving.
  """

  def input_fn():
    """Serving input function that reads raw data and applies transforms."""
    raw_placeholder_spec = RAW_DATA_METADATA.schema.as_batched_placeholders()
    # remove label key that is not going to be available at seving
    raw_placeholder_spec.pop(LABEL_KEY)

    # we are defining the feature_column (raw_featutes) and the tensor
    # (receiver_tensors) for the raw data
    raw_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        raw_placeholder_spec)
    raw_features, receiver_tensors , _ = raw_input_fn()

    # we are tranforming the raw_features with the graph written by
    # preprocess.py to transform_fn_io.TRANSFORM_FN_DIR and that was used to
    # write the tf records. This helps avoiding training/serving skew

    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            os.path.join(tft_working_dir, transform_fn_io.TRANSFORM_FN_DIR),
            raw_features))

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, receiver_tensors)

  return input_fn


def build_estimator(config, tft_working_dir, embedding_size, hidden_units):
  """Build a estimator.

  Args:
    config: tensorflow.python.estimator.run_config.RunConfig defining the
      runtime environment for the estimator (including model_dir).

    tft_working_dir : Directory pointed from the tf transform pipeline

    embedding_size : Dimension of the embedding for the feature columns
      Channel

    hidden_units : [int], the layer sizes of the DNN (input layer first)


  Returns:
    A DNNCombinedLinearClassifier

  Raises:
    LookupError is the transformed_features are not consistent with
      input_metadata.py

  """
  transformed_metadata = metadata_io.read_metadata(
      os.path.join(
          tft_working_dir, transform_fn_io.TRANSFORMED_METADATA_DIR))
  transformed_features = transformed_metadata.schema.as_feature_spec()


  transformed_features.pop(LABEL_KEY)

  real_valued_columns = {}
  categorical_valued_columns = {}
  for key, tensor in transformed_features.items():
    # Separate features by numeric or categorical
    if key in STRING_TO_INT_FEATURE_KEYS:
      if not key in CATEGORICAL_FEATURE_KEYS_TO_BE_REMOVED:
        categorical_valued_columns[key] = tf.feature_column.categorical_column_with_identity(
            key=key,
            num_buckets=vocabulary_size_by_name(tft_working_dir, key)
        )

    elif key in HASH_STRING_FEATURE_KEYS:
      if not key in CATEGORICAL_FEATURE_KEYS_TO_BE_REMOVED:
        categorical_valued_columns[key] = tf.feature_column.categorical_column_with_identity(
            key=key,
            num_buckets=HASH_STRING_FEATURE_KEYS[key]
        )
    elif key in NUMERIC_FEATURE_KEYS:
      if not key in NUMERIC_FEATURE_KEYS_TO_BE_REMOVED:
        real_valued_columns[key] = tf.feature_column.numeric_column(
            key, shape=())

    elif (
        key.endswith('_bucketized') and
        key.replace('_bucketized', '') in TO_BE_BUCKETIZED_FEATURE):
      categorical_valued_columns[key] = tf.feature_column.categorical_column_with_identity(
          key=key,
          num_buckets=TO_BE_BUCKETIZED_FEATURE[key.replace('_bucketized', '')]
      )
    else:
      raise LookupError(
          ('The couple (%s,%s) is not consistent with ',
          'input_metadata.py' % (key, tensor)))


  # creating a new categorical features by crossing
  categorical_valued_columns.update(
      {'education_x_occupation' : tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=int(1e4)),
      'age_bucketized_x_race_x_occupation' : tf.feature_column.crossed_column(
          ['age_bucketized', 'race', 'occupation'], hash_bucket_size=int(1e6)),
      'native_country_x_occupation' : tf.feature_column.crossed_column(
          ['native_country', 'occupation'], hash_bucket_size=int(1e4))
       }
  )

  # creating new numeric features from categorical features
  real_valued_columns.update(
      {
       # Use indicator columns for low dimensional vocabularies
      'workclass_indicator' : tf.feature_column.indicator_column(
          categorical_valued_columns['workclass']),
      'education_indicator' : tf.feature_column.indicator_column(
          categorical_valued_columns['education']),
      'marital_status_indicator' : tf.feature_column.indicator_column(
          categorical_valued_columns['marital_status']),
      'gender_indicator' : tf.feature_column.indicator_column(
          categorical_valued_columns['gender']),
      'relationship_indicator' : tf.feature_column.indicator_column(
          categorical_valued_columns['relationship']),
      'race_indicator' : tf.feature_column.indicator_column(
          categorical_valued_columns['race']),
      # Use embedding columns for high dimensional vocabularies
      'native_country_embedding' : tf.feature_column.embedding_column(
          categorical_valued_columns['native_country'],
          dimension=embedding_size),
      'occupation_embedding' : tf.feature_column.embedding_column(
          categorical_valued_columns['occupation'], dimension=embedding_size),
  })

  return tf.estimator.DNNLinearCombinedClassifier(
      config=config,
      linear_feature_columns=categorical_valued_columns.values(),
      dnn_feature_columns=real_valued_columns.values(),
      dnn_hidden_units=hidden_units or [100, 70, 50, 25]
  )
