# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Schema and tranform definition for the Movielens dataset."""

import hashlib
import numpy as np
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import impl
from tensorflow_transform.tf_metadata import dataset_schema

# Columns of the input file movies.csv.
MOVIE_COLUMNS = ['movie_id', 'title', 'genres']

# Columns for the input file ratings.csv."""
RATING_COLUMNS = ['user_id', 'movie_id', 'rating', 'timestamp']


# Names of feature columns associated with the `Query`. These are the features
# typically included in a recommendation request. In the case of movielens,
# query contains just data about the user. In other applications, there
# could be additional dimensions such as context (i.e. device, time of day, etc)

# The user id.
QUERY_USER_ID = 'query_user_id'
# The ids of movies rated by the user.
QUERY_RATED_MOVIE_IDS = 'query_rated_movie_ids'
# The scores on the rated movies given by the user.
QUERY_RATED_MOVIE_SCORES = 'query_rated_movie_scores'
# The set of genres of the rated movies.
QUERY_RATED_GENRE_IDS = 'query_rated_genre_ids'
# The number of times the user rated each genre.
QUERY_RATED_GENRE_FREQS = 'query_rated_genre_freqs'
# The average rating on each genre.
QUERY_RATED_GENRE_AVG_SCORES = 'query_rated_genre_avg_scores'


# Names of feature columns associated with the `Candidate`. These features
# are used to match a candidate against the query.

# The id of the candidate movie.
CANDIDATE_MOVIE_ID = 'cand_movie_id'
# The set of genres of the candidate movie.
CANDIDATE_GENRE_IDS = 'cand_genre_ids'
# Movie ids used to rank against the target movie. These ranking candidate movie
# ids are used in evaluation only.
RANKING_CANDIDATE_MOVIE_IDS = 'ranking_candidate_movie_ids'

# Names of feature columns defining the label(s), which indicates how well
# a candidate matches a query. There could be multiple labels in each instance.
# Eg. We could have one label for the rating score and another label for the
# number of times a user has watched the movie.
LABEL_RATING_SCORE = 'label_rating_score'


# Each training example consists of a query and a candidate with their
# respective features, as well as one or more labels.
EXAMPLE_COLUMNS = [
    QUERY_USER_ID,
    QUERY_RATED_MOVIE_IDS,
    QUERY_RATED_MOVIE_SCORES,
    QUERY_RATED_GENRE_IDS,
    QUERY_RATED_GENRE_FREQS,
    QUERY_RATED_GENRE_AVG_SCORES,
    CANDIDATE_MOVIE_ID,
    CANDIDATE_GENRE_IDS,
    RANKING_CANDIDATE_MOVIE_IDS,
    LABEL_RATING_SCORE,
]


def _hash_fingerprint(user_id, partition_random_seed):
  """Convert user_id to an MD5 hashed integer.

  The hashed value is based on the input of user_id + partition_random_seed so
  that the output is deterministic for a fixed partition_random_seed and people
  still have the option to partition in a different way by using a different
  seed.

  Args:
    user_id: an integer user id.
    partition_random_seed: partitioning seed so we can preserve consistent
    partitions across runs.
  Returns:
    An MD5 hashed value encoded as integer.
  """
  m = hashlib.md5(str(user_id + partition_random_seed))
  return int(m.hexdigest(), 16)


def partition_fn(user_id, partition_random_seed, percent_eval):
  """Partition data to train and eval set.

  To generate an unskewed partition that is deterministic, we use
  hash_fingerprint(user_id, partition_random_seed) % 100.

  Args:
    user_id: an integer user id.
    partition_random_seed: partitioning seed so we can preserve consistent
    partitions across runs.
    percent_eval: percentage of the data to use as the eval set.
  Returns:
    Either 0 or 1.
  """
  hash_value = _hash_fingerprint(user_id, partition_random_seed) % 100
  return 0 if hash_value >= percent_eval else 1


def create_random_movie_samples(all_movies, movies_to_exclude,
                                num_movies_to_sample, random_seed):
  """Create random samples of movies excluding "movies_to_exclude" list.

  Args:
    all_movies: a list of integer movie ids.
    movies_to_exclude: a list of integer movie ids to exclude.
    num_movies_to_sample: number of movie ids to sample.
    random_seed: random seed for numpy random number generator.
  Returns:
    A list of integer movie ids.
  """

  candidate_movie_ids = set(all_movies).difference(movies_to_exclude)
  # Sort the set candidate_movie_ids first to make results reproducible for
  # expectation test.
  return np.random.RandomState(random_seed).choice(
      sorted(candidate_movie_ids), num_movies_to_sample,
      replace=False).tolist()


def _make_schema(columns, types, default_values):
  """Input schema definition.

  Args:
    columns: column names for fields appearing in input.
    types: column types for fields appearing in input.
    default_values: default values for fields appearing in input.
  Returns:
    feature_set dictionary of string to *Feature.
  """
  result = {}
  assert len(columns) == len(types)
  assert len(columns) == len(default_values)
  for c, t, v in zip(columns, types, default_values):
    if isinstance(t, list):
      result[c] = tf.VarLenFeature(dtype=t[0])
    else:
      result[c] = tf.FixedLenFeature(shape=[], dtype=t, default_value=v)
  return dataset_schema.from_feature_spec(result)


def make_ratings_schema():
  return _make_schema(RATING_COLUMNS,
                      [tf.int64, tf.string, tf.float32, tf.int64],
                      [-1, '', 0.0, -1])


def make_movies_schema():
  return _make_schema(MOVIE_COLUMNS,
                      [tf.string, tf.string, [tf.string]],
                      ['', '', None])


def make_examples_schema():
  return _make_schema(EXAMPLE_COLUMNS, [
      tf.int64, [tf.string], [tf.float32], [tf.string], [tf.float32],
      [tf.float32], [tf.string], [tf.string], [tf.string], tf.float32
  ], [-1, None, None, None, None, None, -1, None, None, 0.0])


def make_prediction_schema():
  prediction_columns = [column for column in EXAMPLE_COLUMNS
                        if column != LABEL_RATING_SCORE]
  return _make_schema(prediction_columns, [
      tf.int64, [tf.string], [tf.float32], [tf.string], [tf.float32],
      [tf.float32], [tf.string], [tf.string], [tf.string]
  ], [-1, None, None, None, None, None, -1, None, None])


def make_preprocessing_fn():
  """Creates a preprocessing function for movielens.

  Returns:
    A preprocessing function.
  """
  def preprocessing_fn(inputs):
    """User defined preprocessing function for movielens columns.

    Args:
      inputs: a `dict` that maps EXAMPLE_COLUMNS to the corresponding
        Tensor/SparseTensor.
    Returns:
      A `dict` that maps EXAMPLE_COLUMNS to the transformed Tensor/SparseTensor.
    """
    result = {column_name: inputs[column_name]
              for column_name in EXAMPLE_COLUMNS}

    rating_max = tft.max(inputs[QUERY_RATED_MOVIE_SCORES].values)

    rating_min = tft.min(inputs[QUERY_RATED_MOVIE_SCORES].values)

    def scale_sparse_values(x, min_value, max_value):
      """0-1 normalization of the values of a SparseTensor.

      Args:
        x: a input sparse tensor.
        min_value: minimum value for x.values.
        max_value: maximum value for x.values.
      Returns:
        A sparse tensor y such as that y.values is the result of
        0-1 normalization of x.values.
      """
      scaled_values = (x.values - min_value) / (max_value - min_value)
      return tf.SparseTensor(indices=x.indices, values=scaled_values,
                             dense_shape=x.dense_shape)

    result[QUERY_RATED_MOVIE_SCORES] = scale_sparse_values(
        inputs[QUERY_RATED_MOVIE_SCORES],
        rating_min, rating_max)

    genre_vocab = tft.uniques(tf.concat(
        [inputs[QUERY_RATED_GENRE_IDS].values,
         inputs[CANDIDATE_GENRE_IDS].values], 0))

    movie_vocab = tft.uniques(tf.concat(
        [inputs[QUERY_RATED_MOVIE_IDS].values,
         inputs[CANDIDATE_MOVIE_ID].values,
         inputs[RANKING_CANDIDATE_MOVIE_IDS].values], 0))

    def map_to_int(x, vocabulary_or_file):
      """Maps string tensor into indexes using vocab.

      Args:
        x : a Tensor/SparseTensor of string.
        vocabulary_or_file: a Tensor/SparseTensor containing unique string
          values within x or a single value for the file where the vocabulary
          is stored.

      Returns:
        A Tensor/SparseTensor of indexes (int) of the same shape as x.
      """
      # TODO(b/62489180): Remove this workaround once TFT 0.2.0 is released.
      if hasattr(impl,
                 '_asset_files_supported') and impl._asset_files_supported():  # pylint: disable=protected-access
        table = tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=vocabulary_or_file, num_oov_buckets=1)
      else:
        table = tf.contrib.lookup.string_to_index_table_from_tensor(
            mapping=vocabulary_or_file, num_oov_buckets=1)
      return table.lookup(x)

    result[QUERY_RATED_GENRE_IDS] = tft.apply_function(
        map_to_int, inputs[QUERY_RATED_GENRE_IDS], genre_vocab)

    result[CANDIDATE_GENRE_IDS] = tft.apply_function(
        map_to_int, inputs[CANDIDATE_GENRE_IDS], genre_vocab)

    result[QUERY_RATED_MOVIE_IDS] = tft.apply_function(
        map_to_int, inputs[QUERY_RATED_MOVIE_IDS], movie_vocab)

    result[CANDIDATE_MOVIE_ID] = tft.apply_function(
        map_to_int, inputs[CANDIDATE_MOVIE_ID], movie_vocab)

    result[RANKING_CANDIDATE_MOVIE_IDS] = tft.apply_function(
        map_to_int, inputs[RANKING_CANDIDATE_MOVIE_IDS], movie_vocab)

    return result

  return preprocessing_fn
