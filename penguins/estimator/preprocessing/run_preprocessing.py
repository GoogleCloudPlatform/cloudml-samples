# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Runs the preprocessing job to produce records for training."""

import argparse
import ConfigParser
import logging
import os
import sys

import apache_beam as beam

from preprocessing import preprocess


def _parse_arguments(argv):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description='Runs preprocessing on census train data.')
    parser.add_argument(
        '--project_id',
        required=True,
        help='Name of the project.')
    parser.add_argument(
        '--job_name',
        required=False,
        help='Name of the dataflow job.')
    parser.add_argument(
        '--job_dir',
        required=True,
        help='Directory to write outputs.')
    parser.add_argument(
        '--cloud',
        default=False,
        action='store_true',
        help='Run preprocessing on the cloud.')
    parser.add_argument(
        '--input_data',
        required=True,
        help='Path to input data.')
    args, _ = parser.parse_known_args(args=argv[1:])
    return args


def _parse_config(env, config_file_path):
    """Parses configuration file.

    Args:
      env: The environment in which the preprocessing job will be run.
      config_file_path: Path to the configuration file to be parsed.

    Returns:
      A dictionary containing the parsed runtime config.
    """
    config = ConfigParser.ConfigParser()
    config.read(config_file_path)
    return dict(config.items(env))


def _set_logging(log_level):
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))


def main():
    """Configures pipeline and spawns preprocessing job."""

    args = _parse_arguments(sys.argv)
    config_path = os.path.abspath(
        os.path.join(__file__, os.pardir, 'preprocessing_config.ini'))
    config = _parse_config('CLOUD' if args.cloud else 'LOCAL',
                           config_path)
    ml_project = args.project_id
    options = {'project': ml_project}

    if args.cloud:
        if not args.job_name:
            raise ValueError('Job name must be specified for cloud runs.')
        options.update({
            'job_name': args.job_name,
            'num_workers': int(config.get('num_workers')),
            'max_num_workers': int(config.get('max_num_workers')),
            'staging_location': os.path.join(args.job_dir, 'staging'),
            'temp_location': os.path.join(args.job_dir, 'tmp'),
            'region': config.get('region'),
            'setup_file': os.path.abspath(
                os.path.join(__file__, '../..', 'dataflow_setup.py')),
        })
    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)
    _set_logging(config.get('log_level'))

    with beam.Pipeline(config.get('runner'), options=pipeline_options) as p:
        preprocess.run(p, args.input_data, args.job_dir)


if __name__ == '__main__':
    main()
