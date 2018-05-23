# Copyright 2018 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import print_function

import StringIO
import argparse
import ftplib
import multiprocessing as mp
import os
import sys
import tempfile
import tensorflow as tf
import zlib


DEFAULT_TEMP_DIR = os.path.join(
    tempfile.gettempdir(), 'cloudml-samples', 'molecules')

FTP_SERVER = 'ftp.ncbi.nlm.nih.gov'
FTP_USER = 'anonymous'
FTP_PASS = 'guest'
FTP_PATH = 'pubchem/Compound_3D/01_conf_per_cmpd/SDF'

# Good for debugging
FORCE_DISABLE_MULTIPROCESSING = False


def _function_wrapper(args_tuple):
  """Function wrapper to call from multiprocessing."""
  function, args = args_tuple
  try:
    return function(*args)
  except KeyboardInterrupt:
    pass


def parallel_map(function, iterable):
  """Calls a function for every element in an iterable using multiple cores."""
  if FORCE_DISABLE_MULTIPROCESSING:
    return [function(*args) for args in iterable]

  num_threads = mp.cpu_count() * 2
  pool = mp.Pool(processes=num_threads)
  p = pool.map_async(_function_wrapper, ((function, args) for args in iterable))
  try:
    results = p.get()
  except KeyboardInterrupt:
    raise
  pool.close()
  return results


def extract_data_file(ftp_path, data_dir):
  """Function to extract a single PubChem data file."""
  basename = os.path.basename(ftp_path)
  sdf_file = os.path.join(data_dir, os.path.splitext(basename)[0])

  if not tf.gfile.Exists(sdf_file):
    print('Extracting data file: {}'.format(sdf_file))
    # The `ftp` object cannot be pickled for multithreading, so we open a
    # new connection here
    memfile = StringIO.StringIO()
    ftp = ftplib.FTP(FTP_SERVER, FTP_USER, FTP_PASS)
    ftp.retrbinary('RETR ' + ftp_path, memfile.write)
    ftp.quit()

    memfile.seek(0)
    with tf.gfile.Open(sdf_file, 'w') as f:
      gzip_wbits_format = zlib.MAX_WBITS | 16
      contents = zlib.decompress(memfile.getvalue(), gzip_wbits_format)
      f.write(contents)

  else:
    print('Found data file: {}'.format(sdf_file))


def run(total_data_files, data_dir):
  """Extracts the specified number of data files in parallel."""
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  # Get available data files
  print('Listing data files from: {}/{}'.format(FTP_SERVER, FTP_PATH))
  ftp = ftplib.FTP(FTP_SERVER, FTP_USER, FTP_PASS)
  ftp_files = ftp.nlst(FTP_PATH)
  ftp.quit()

  # Extract data files in parallel
  if total_data_files is None:
    total_data_files = len(ftp_files)
  assert total_data_files >= 1
  print('Found {} files, using {}'.format(len(ftp_files), total_data_files))
  ftp_files = ftp_files[:total_data_files]
  parallel_map(
      extract_data_file, ((ftp_path, data_dir)
        for i, ftp_path in enumerate(ftp_files)))


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--data-dir',
                      type=str,
                      default=os.path.join(DEFAULT_TEMP_DIR, 'data'),
                      help='Directory to store the data files to. '
                           'This can be a Google Cloud Storage path.')
  parser.add_argument('--total-data-files',
                      type=int,
                      default=5,
                      help='Total number of data files to use, '
                           'set to `-1` to use all data files. '
                           'Each data file contains 25,000 molecules')
  args = parser.parse_args()

  if args.total_data_files < 1 and args.total_data_files != -1:
    print('Error: --total-data-files must be >= 1 or -1 to use all')
    sys.exit(1)

  total_data_files = args.total_data_files
  if args.total_data_files == -1:
    total_data_files = None

  run(total_data_files, args.data_dir)
