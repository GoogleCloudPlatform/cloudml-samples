#!/usr/bin/env python

# Copyright 2019 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This tool downloads SDF files from an FTP source.

from __future__ import absolute_import

import argparse
import ftplib
import multiprocessing as mp
import os
import re
import signal
import tempfile
import tensorflow as tf
import zlib
from io import BytesIO


# Regular expressions to parse an FTP URI.
_USER_RE = r'''(?P<user>[^:@]+|'[^']+'|"[^"]+")'''
_PASSWORD_RE = r'''(?P<password>[^@]+|'[^']+'|"[^"]+")'''
_CREDS_RE = r'{}(?::{})?'.format(_USER_RE, _PASSWORD_RE)
FTP_RE = re.compile(r'^ftp://(?:{}@)?(?P<abs_path>.*)$'.format(_CREDS_RE))

# Good for debugging.
FORCE_DISABLE_MULTIPROCESSING = False


def _function_wrapper(args_tuple):
  """Function wrapper to call from multiprocessing."""
  function, args = args_tuple
  return function(*args)


def parallel_map(function, iterable):
  """Calls a function for every element in an iterable using multiple cores."""
  if FORCE_DISABLE_MULTIPROCESSING:
    return [function(*args) for args in iterable]

  original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
  num_threads = mp.cpu_count() * 2
  pool = mp.Pool(processes=num_threads)
  signal.signal(signal.SIGINT, original_sigint_handler)

  p = pool.map_async(_function_wrapper, ((function, args) for args in iterable))
  try:
    results = p.get(0xFFFFFFFF)
  except KeyboardInterrupt:
    pool.terminate()
    raise
  pool.close()
  return results


def extract_data_file(ftp_file, data_dir):
  """Function to extract a single PubChem data file."""
  user = ftp_file['user']
  password = ftp_file['password']
  server = ftp_file['server']
  path = ftp_file['path']
  basename = os.path.basename(path)
  sdf_file = os.path.join(data_dir, os.path.splitext(basename)[0])

  if not tf.gfile.Exists(sdf_file):
    # The `ftp` object cannot be pickled for multithreading, so we open a
    # new connection here
    memfile = BytesIO()
    ftp = ftplib.FTP(server, user, password)
    ftp.retrbinary('RETR ' + path, memfile.write)
    ftp.quit()

    memfile.seek(0)
    with tf.gfile.Open(sdf_file, 'w') as f:
      gzip_wbits_format = zlib.MAX_WBITS | 16
      contents = zlib.decompress(memfile.getvalue(), gzip_wbits_format)
      f.write(contents)
    print('Extracted {}'.format(sdf_file))

  else:
    print('Found {}'.format(sdf_file))


def run(data_sources, filter_regex, max_data_files, data_dir):
  """Extracts the specified number of data files in parallel."""
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  # Get available data files
  filter_re = re.compile(filter_regex)
  ftp_files = []
  for source in data_sources:
    m = FTP_RE.search(source)
    if not m:
      raise ValueError('malformed FTP URI')
    user = m.group('user') or 'anonymous'
    password = m.group('password') or 'guest'
    server, path_dir = m.group('abs_path').split('/', 1)
    uri_prefix = 'ftp://{}:{}@{}/'.format(user, password, server)

    ftp = ftplib.FTP(server, user, password)
    ftp_files += [{
        'user': user,
        'password': password,
        'server': server,
        'path': path,
      } for path in ftp.nlst(path_dir)
      if filter_re.search(uri_prefix + path)]
    ftp.quit()

  # Extract data files in parallel
  if not max_data_files:
    max_data_files = len(ftp_files)
  assert max_data_files >= 1
  print('Found {} files, using {}'.format(len(ftp_files), max_data_files))
  ftp_files = ftp_files[:max_data_files]
  print('Extracting data files...')
  parallel_map(
      extract_data_file, ((ftp_file, data_dir) for ftp_file in ftp_files))


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--work-dir',
      required=True,
      help='Directory for staging and working files. '
           'This can be a Google Cloud Storage path.')

  parser.add_argument(
      '--data-sources',
      nargs='+',
      default=['ftp://anonymous:guest@ftp.ncbi.nlm.nih.gov/'
               'pubchem/Compound_3D/01_conf_per_cmpd/SDF'],
      help='Data source location where SDF file(s) are stored. '
           'Paths can be local, ftp://<path>, or gcs://<path>. '
           'Examples: '
           'ftp://hostname/path '
           'ftp://username:password@hostname/path')

  parser.add_argument(
      '--filter-regex',
      default=r'\.sdf',
      help='Regular expression to filter which files to use. '
           'The regular expression will be searched on the full absolute path. '
           'Every match will be kept.')

  parser.add_argument(
      '--max-data-files',
      type=int,
      required=True,
      help='Maximum number of data files for every file pattern expansion. '
           'Set to -1 to use all files.')

  args = parser.parse_args()

  max_data_files = args.max_data_files
  if args.max_data_files == -1:
    max_data_files = None

  data_dir = os.path.join(args.work_dir, 'data')
  run(args.data_sources, args.filter_regex, max_data_files, data_dir)
