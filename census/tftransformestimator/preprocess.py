from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

with warnings.catch_warnings():
  warnings.filterwarnings("ignore", category=DeprecationWarning)

  import argparse
  import os
  import six


  from input_metadata import HASH_STRING_FEATURE_KEYS
  from input_metadata import LABEL_KEY
  from input_metadata import NUMERIC_FEATURE_KEYS
  from input_metadata import ORDERED_COLUMNS
  from input_metadata import RAW_DATA_METADATA
  from input_metadata import STRING_TO_INT_FEATURE_KEYS
  from input_metadata import TO_BE_BUCKETIZED_FEATURE

  import apache_beam as beam
  from apache_beam.io import textio
  from apache_beam.io import tfrecordio
  import tensorflow as tf
  import tensorflow_transform as tft
  from tensorflow.contrib import lookup
  from tensorflow_transform.beam import impl as beam_impl
  from tensorflow_transform.beam.tft_beam_io import transform_fn_io
  from tensorflow_transform.coders import csv_coder
  from tensorflow_transform.coders import example_proto_coder
  from tensorflow_transform.saved import saved_transform_io
  from tensorflow_transform.tf_metadata import metadata_io



# Functions for preprocessing
def transform_data(train_data_file,
                   test_data_file,
                   working_dir,
                   root_train_data_out,
                   root_test_data_out,
                   runner='DirectRunner',
                   pipeline_options={}):
  """Transform the data and write out as a TFRecord of Example protos.
  Read in the data using the CSV reader, and transform it using a
  preprocessing pipeline that scales numeric data and converts categorical data
  from strings to int64 values indices, by creating a vocabulary for each
  category.
  Args:
    train_data_file: File containing training data
    test_data_file: File containing test data
    working_dir: Directory to write transformed data and metadata to
    root_train_data_out: Root of file containing transform training data
    root_test_data_out: Root of file containing transform test data
    runner: Beam runner ('DirectRunner' or 'DataflowRunner')
    pipeline_options: dictionary with Dataflow configuration
  """

  def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}

    # Scale numeric columns to have range [0, 1].
    for key in NUMERIC_FEATURE_KEYS:
      outputs[key] = tft.scale_to_0_1(inputs[key])

    # bucketize numeric columns
    for key in TO_BE_BUCKETIZED_FEATURE:
      outputs[key+'_bucketized'] = tft.bucketize(
          inputs[key],
          TO_BE_BUCKETIZED_FEATURE[key]
      )


    # For categorical columns with a small vocabulary
    for key in STRING_TO_INT_FEATURE_KEYS:
      outputs[key] = tft.string_to_int(
          inputs[key],
          vocab_filename=key)

    for key in HASH_STRING_FEATURE_KEYS:
      outputs[key] = tft.hash_strings(inputs[key], HASH_STRING_FEATURE_KEYS[key])

    # For the label column we transform it either 0 or 1 if there are row leads
    def convert_label(label):
      """Parses a string tensor into the label tensor
      Args:
        label_string_tensor: Tensor of dtype string. Result of parsing the
        CSV column specified by LABEL_COLUMN
      Returns:
        A Tensor of the same shape as label_string_tensor, should return
        an int64 Tensor representing the label index for classification tasks
      """
      table = lookup.index_table_from_tensor(['<=50K', '>50K'])
      return table.lookup(label)

    outputs[LABEL_KEY] = tft.apply_function(convert_label, inputs[LABEL_KEY])
    return outputs

  def fix_comma_and_filter_third_column(line):
    cols = line.replace(', ', ',').split(',')
    return ','.join(cols[0:2] + cols[3:])

  # The "with" block will create a pipeline, and run that pipeline at the exit
  # of the block.
  with beam.Pipeline(runner=runner,
                     options=beam.pipeline.PipelineOptions(flags=[],
                                                           **pipeline_options)) as pipeline:
    with beam_impl.Context(pipeline_options['temp_location']):
      # Create a coder to read the census data with the schema.  To do this we
      # need to list all columns in order since the schema doesn't specify the
      # order of columns in the csv.

      converter = csv_coder.CsvCoder(ORDERED_COLUMNS, RAW_DATA_METADATA.schema)

      # Read in raw data and convert using CSV converter.  Note that we apply
      # some Beam transformations here, which will not be encoded in the TF
      # graph since we don't do the from within tf.Transform's methods
      # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
      # to get data into a format that the CSV converter can read, in particular
      # removing empty lines and removing spaces after commas.




      raw_data = (
          pipeline
          | 'ReadTrainData' >> textio.ReadFromText(train_data_file)
          | 'FilterTrainData' >> beam.Filter(lambda line: line)
          | 'FixCommasAndRemoveFiledTrainData' >> beam.Map(
              fix_comma_and_filter_third_column)
          | 'DecodeTrainData' >> beam.Map(converter.decode))

      # Combine data and schema into a dataset tuple.  Note that we already used
      # the schema to read the CSV data, but we also need it to interpret
      # raw_data.
      raw_dataset = (raw_data, RAW_DATA_METADATA)
      transformed_dataset, transform_fn = (
          raw_dataset | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data, transformed_metadata = transformed_dataset

      _ = transformed_data | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(
          os.path.join(working_dir, root_train_data_out),
          coder=example_proto_coder.ExampleProtoCoder(
              transformed_metadata.schema))

      # Now apply transform function to test data.  In this case we also remove
      # the header line from the CSV file and the trailing period at the end of
      # each line.
      raw_test_data = (
          pipeline
          | 'ReadTestData' >> textio.ReadFromText(test_data_file)
          | 'FilterTestData' >> beam.Filter(lambda line: line)
          | 'FixCommasAndRemoveFiledTestData' >> beam.Map(
              fix_comma_and_filter_third_column)
          | 'DecodeTestData' >> beam.Map(converter.decode))

      raw_test_dataset = (raw_test_data, RAW_DATA_METADATA)

      transformed_test_dataset = (
          (raw_test_dataset, transform_fn) | beam_impl.TransformDataset())
      # Don't need transformed data schema, it's the same as before.
      transformed_test_data, _ = transformed_test_dataset

      _ = transformed_test_data | 'WriteTestData' >> tfrecordio.WriteToTFRecord(
          os.path.join(working_dir, root_test_data_out),
          coder=example_proto_coder.ExampleProtoCoder(
              transformed_metadata.schema))

      # Will write a SavedModel and metadata to two subdirectories of
      # working_dir, given by transform_fn_io.TRANSFORM_FN_DIR and
      # transform_fn_io.TRANSFORMED_METADATA_DIR respectively.
      _ = (
          transform_fn
          | 'WriteTransformFn' >>
          transform_fn_io.WriteTransformFn(working_dir))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--train-data-file',
    help='Path to training data',
    required=True
  )
  parser.add_argument(
      '--test-data-file',
      help='Path to test data',
      required=True
  )
  parser.add_argument(
      '--root-train-data-out',
      help='Root for files with train data',
      required=True
  )
  parser.add_argument(
      '--root-test-data-out',
      help='Root for files with test data',
      required=True
  )
  parser.add_argument(
      '--working-dir',
      help='Path to the directory where transformed data are written',
      required=True
  )
  parser.add_argument(
      '--runner',
      help="Apache Beam runner: DirectRunner or DataflowRunner",
      choices=['DirectRunner','DataflowRunner'],
      default='DirectRunner'
  )
  parser.add_argument(
      '--job-name',
      help="Job name on Cloud Dataflow",
      type=str
  )
  parser.add_argument(
      '--project',
      help="GCP's project",
      type=str
  )
  parser.add_argument(
      '--region',
      help="GPS's region",
      type=str
  )
  parser.add_argument(
      '--staging-location',
      help='Path to staging location',
      type=str
  )
  parser.add_argument(
      '--temp-location',
      help='Path to temp location',
      type=str
  )

  parser.add_argument(
        '--requirements-file',
        help='Path to requirements python package',
        type=str
  )

  parser.add_argument(
        '--extra-package',
        help='Paths to python package that do not have source dists',
        nargs='+',
        type=str
  )

  args = parser.parse_args()


  if args.runner == 'DataflowRunner':
    if args.staging_location is None:
      args.staging_location = os.path.join(args.working_dir,
                                           'tmp',
                                           'staging')

    if args.temp_location is None:
      args.temp_location = os.path.join(args.working_dir,
                                           'tmp')

    if (args.job_name is None or
        args.project is None or
        args.region is None):
      parser.error(('--runner DataflowRunner requires --job-name, '
                    '--project, --region, --staging-location',
                    '--temp-location')
      )



    options={
      'staging_location': args.staging_location,
      'temp_location': args.temp_location,
      'job_name': args.job_name,
      'project': args.project,
      'region' : args.region,
      'teardown_policy': 'TEARDOWN_ALWAYS',
      'no_save_main_session': True
    }

  else:
    options = {'temp_location': args.temp_location}

  if args.requirements_file:
    options['requirements_file'] = args.requirements_file

  if args.extra_package:
    options['extra_package'] = args.extra_package

  transform_data(train_data_file=args.train_data_file,
                 test_data_file=args.test_data_file,
                 working_dir=args.working_dir,
                 root_train_data_out=args.root_train_data_out,
                 root_test_data_out=args.root_test_data_out,
                 runner=args.runner,
                 pipeline_options=options)


if __name__ == '__main__':
    main()
