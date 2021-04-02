# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os
import subprocess

import numpy as np

from tensorflow.python.keras.preprocessing.sequence import _remove_long_seq
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

IMDB_FILE = 'imdb.npz'
INDEX_FILE = 'imdb_word_index.json'
SENTENCE_SIZE = 256
WORKING_DIR = os.getcwd()

def _load_data(path='imdb.npz',
               num_words=None,
               skip_top=0,
               maxlen=None,
               seed=113,
               start_char=1,
               oov_char=2,
               index_from=3,
               **kwargs):
  """Helper function.

  Loads the IMDB dataset in npz format.

    IMDB dataset contains the text of 50,000 movie reviews from the
    Internet Movie Database. These are split into 25,000 reviews for training
    and 25,000 reviews for testing. The training and testing sets are balanced,
    meaning they contain an equal number of positive and negative reviews.
    This function handles the pre-process file in .npz format.
    The text of reviews have been converted to integers, where each integer
    represents a specific word in a dictionary.

  Args:
    path: Where file is located.
    num_words: max number of words to include. Words are ranked by how often
          they occur (in the training set) and only the most frequent words are
          kept
    skip_top: skip the top N most frequently occurring words (which may not
          be informative).
    maxlen: sequences longer than this will be filtered out.
    seed: random seed for sample shuffling.
    start_char: The start of a sequence will be marked with this character.
          Set to 1 because 0 is usually the padding character.
    oov_char: words that were cut out because of the `num_words` or
          `skip_top` limit will be replaced with this character.
    index_from: index actual words with this index and higher.
    **kwargs: Used for backwards compatibility.

  Returns:
    A tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  Raises:
    ValueError: In case path is not defined.
    ValueError: In case `maxlen` is so low
            that no input sequence could be kept.

  Note that the 'out of vocabulary' character is only used for
  words that were present in the training set but are not included
  because they're not making the `num_words` cut here.
  Words that were not seen in the training set but are in the test set
  have simply been skipped.
  """
  if not path:
    raise ValueError('No training file defined')

  if path.startswith('gs://'):
    download_files_from_gcs(path, destination=IMDB_FILE)
    path = IMDB_FILE
  with np.load(path) as f:
    x_train, labels_train = f['x_train'], f['y_train']
    x_test, labels_test = f['x_test'], f['y_test']

  np.random.seed(seed)
  indices = np.arange(len(x_train))
  np.random.shuffle(indices)
  x_train = x_train[indices]
  labels_train = labels_train[indices]

  indices = np.arange(len(x_test))
  np.random.shuffle(indices)
  x_test = x_test[indices]
  labels_test = labels_test[indices]

  xs = np.concatenate([x_train, x_test])
  labels = np.concatenate([labels_train, labels_test])

  if start_char is not None:
    xs = [[start_char] + [w + index_from for w in x] for x in xs]
  elif index_from:
    xs = [[w + index_from for w in x] for x in xs]

  if maxlen:
    xs, labels = _remove_long_seq(maxlen, xs, labels)
    if not xs:
      raise ValueError('After filtering for sequences shorter than maxlen=' +
                       str(maxlen) + ', no sequence was kept. '
                       'Increase maxlen.')
  if not num_words:
    num_words = max([max(x) for x in xs])

  # By convention, use 2 as OOV word
  # reserve 'index_from' (=3 by default) characters:
  # 0 (padding), 1 (start), 2 (OOV)
  if oov_char is not None:
    xs = [
        [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
    ]
  else:
    xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

  idx = len(x_train)
  x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
  x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
  return (x_train, y_train), (x_test, y_test)


def _decode_review(word_index_file, integer_text):
  """Query a dictionary object that contains the integer to string mapping.

    Example:
    [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173,
    36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4,
    172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447,
    4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87,
    12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18,
    2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33,
    4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22,
    12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256,
    4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2,
    1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486,
    18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28,
    224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472,
    113, 103, 32, 15, 16, 5345, 19, 178, 32]

    "This film was just brilliant casting location scenery story direction
    everyone's really suited the part they played and you could just imagine
    being there robert  is an amazing actor and now the same being director
    father came from the same scottish island as myself so i loved the fact
    there was a real connection with this film the witty remarks throughout
    the film were great it was just brilliant so much that i bought the film as
    soon as it was released for  and would recommend it to everyone to watch
    and the fly fishing was amazing really cried at the end it was so sad
    and you know what they say if you cry at a film it must have been good and
    this definitely was also  to the two little boy's that played the
    of norman and paul they were just brilliant children are often left out of
    the list i think because the stars that play them all grown up are such a
    big profile for the whole film but these children are amazing and should be
    praised for what they have done don't you think the whole story was so
    lovely because it was true and was someone's life after all that was shared
    with us all"

  Args:
    word_index_file: (str) Location of index file.
    integer_text: (Sequence) Sequence of integers.

  Returns:
    A text string.
  """
  word_index = _get_word_index(word_index_file)
  reverse_word_index = dict(
      [(value, key) for (key, value) in word_index.items()])
  return ' '.join([reverse_word_index.get(i, '?') for i in integer_text])


def _get_word_index(path):
  """Gets JSON index file with word information.

  Args:
    path: (str) Location of index file.

  Returns:
    A dictionary with word information.

  Raises:
    ValueError: No index file is defined.
  """
  if not path:
    raise ValueError('No index file defined')
  if path.startswith('gs://'):
    download_files_from_gcs(path, destination=INDEX_FILE)
    path = INDEX_FILE

  with open(path) as f:
    word_index = json.load(f)
  # The first indices are reserved.
  word_index = {k: (v + 3) for k, v in word_index.items()}
  word_index['<PAD>'] = 0
  word_index['<START>'] = 1
  word_index['<UNK>'] = 2  # unknown
  word_index['<UNUSED>'] = 3
  return word_index


def download_files_from_gcs(source, destination):
  """Download files from GCS to a WORKING_DIR/.
  Args:
    source: GCS path to the training data
    destination: GCS path to the validation data.
  Returns:
    A list to the local data paths where the data is downloaded.
  """
  local_file_names = [destination]
  gcs_input_paths = [source]

  # Copy raw files from GCS into local path.
  raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name)
    for local_file_name in local_file_names
    ]
  for i, gcs_input_path in enumerate(gcs_input_paths):
    if gcs_input_path:
      subprocess.check_call(
        ['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])

  return raw_local_files_data_paths


def preprocess(train_data_file, word_index_file, num_words):
  """Loads Numpy file .npz format and process its the data.

  Pad the arrays so they all have the same length, then create an integer
  tensor of shape max_length * num_reviews. Then we use an embedding layer
  capable of handling this shape as the first layer in our network.

  Args:
    train_data_file: (str) Location of file.
    word_index_file: (str) Location of JSON file with index information.
    num_words: (int) Number of words to get from IMDB dataset.

  Returns:
    A tuple of training and test data.
  """
  (train_data, train_labels), (test_data, test_labels) = _load_data(
      path=train_data_file, num_words=num_words)
  word_index = _get_word_index(word_index_file)
  # Standardize the lengths for training.
  train_data = pad_sequences(train_data, value=word_index['<PAD>'],
                             padding='post', maxlen=SENTENCE_SIZE)
  # Standardize the lengths for test.
  test_data = pad_sequences(test_data, value=word_index['<PAD>'],
                            padding='post', maxlen=SENTENCE_SIZE)
  return (train_data, train_labels), (test_data, test_labels)