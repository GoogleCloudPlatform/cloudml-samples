from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

STRING_TO_INT_FEATURE_KEYS = [
    'workclass', 'education', 'marital_status', 'relationship',
    'race', 'gender', 'income_bracket']

HASH_STRING_FEATURE_KEYS = {'occupation' : 100, 'native_country' : 100}


CATEGORICAL_FEATURE_KEYS = list(
    set(STRING_TO_INT_FEATURE_KEYS + HASH_STRING_FEATURE_KEYS.keys()))

CATEGORICAL_FEATURE_KEYS_TO_BE_REMOVED = []

NUMERIC_FEATURE_KEYS = [
    'age', 'education_num', 'capital_gain',
    'capital_loss', 'hours_per_week']

TO_BE_BUCKETIZED_FEATURE = {
  'age' : 10
}

NUMERIC_FEATURE_KEYS_TO_BE_REMOVED = ['age']

LABEL_KEY = 'income_bracket'

ORDERED_COLUMNS = ['age', 'workclass', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'gender',
               'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'income_bracket']


def _create_raw_metadata():
  """Create a DatasetMetadata for the raw data."""
  column_schemas = {
      key: dataset_schema.ColumnSchema(
          tf.string, [], dataset_schema.FixedColumnRepresentation())
      for key in CATEGORICAL_FEATURE_KEYS
  }
  column_schemas.update({
      key: dataset_schema.ColumnSchema(
          tf.float32, [], dataset_schema.FixedColumnRepresentation())
      for key in NUMERIC_FEATURE_KEYS
  })
  column_schemas[LABEL_KEY] = dataset_schema.ColumnSchema(
      tf.string, [], dataset_schema.FixedColumnRepresentation())

  raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema(
      column_schemas))
  return raw_data_metadata

RAW_DATA_METADATA = _create_raw_metadata()
