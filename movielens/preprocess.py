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
"""Movielens Sample Preprocessing Runner."""
import argparse
import datetime
import os
import random
import subprocess
import sys

import apache_beam as beam

from tensorflow_transform import coders as tft_coders
from tensorflow_transform.beam import impl as tft
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.tf_metadata import dataset_metadata

EVAL_TYPES = ['regression', 'ranking']
REGRESSION, RANKING = EVAL_TYPES


def _default_project():
  get_project = ['gcloud', 'config', 'list', 'project',
                 '--format=value(core.project)']

  with open(os.devnull, 'w') as dev_null:
    return subprocess.check_output(get_project, stderr=dev_null).strip()


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments including program name.
  Returns:
    The parsed arguments as returned by argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on the movielens data.')

  parser.add_argument(
      '--project_id', help='The project to which the job will be submitted.')
  parser.add_argument(
      '--cloud', action='store_true', help='Run preprocessing on the cloud.')
  parser.add_argument(
      '--input_dir',
      required=True,
      help='Directory containing the training data.')
  parser.add_argument(
      '--percent_eval',
      type=int,
      required=True,
      help=('Percentage of data to use for evaluation,'
            'a number between 1 and 99.'))
  parser.add_argument(
      '--negative_sample_ratio',
      type=int,
      default=0,
      help=('Ratio of negative samples over observed ratings.'
            'Typically a value between 0 and 10.'))
  parser.add_argument(
      '--negative_sample_label',
      type=float,
      default=1.0,
      help=('Label on negative samples.'))
  parser.add_argument(
      '--eval_score_threshold',
      type=float,
      default=0,
      help=('Threshold on ratings to use for eval.'))
  parser.add_argument(
      '--random_seed',
      type=int,
      default=None,
      help=('Seed for the random number generator'))
  parser.add_argument(
      '--partition_random_seed',
      type=int,
      default=0,
      help='Random seed for train/eval partition')
  parser.add_argument(
      '--output_dir',
      required=True,
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))
  parser.add_argument(
      '--eval_type',
      choices=EVAL_TYPES,
      default=REGRESSION,
      help=('When computing ranking based metrics (precision@K, recall@K), only'
            'evaluate on examples where ratings are higher than'
            'eval_score_threshold.'))
  # Only use this parameter when comparing across datasets.
  parser.add_argument(
      '--num_ranking_candidate_movie_ids',
      type=int,
      default=0,
      help=('(Advanced) Number of unrated movies to rank against the target '
            'movie.'))
  parser.add_argument(
      '--skip_header_lines',
      type=int,
      default=1,
      help='Whether to skip header lines, (0 or 1), defaults to skip.')
  parser.add_argument(
      '--movie_rating_history',
      type=int,
      default=200,
      help=('Number of top ratings to keep for each user from the rating'
            'history, defaults to 200.'))
  parser.add_argument(
      '--runner',
      default=None,
      help=('Specify a runner to execute the pipeline, otherwise rely on '
            'the get_pipeline_name heuristic.'))
  args, _ = parser.parse_known_args(args=argv[1:])
  if args.cloud and not args.project_id:
    args.project_id = _default_project()
  return args


class BuildExampleFn(beam.DoFn):
  """Base class of DoFns for building Movielens Tensorflow Examples.

  The base class only defines a set of string constants defining feature
  column names.
  """

  def __init__(self, random_seed):
    self._random_seed = random_seed

  def _construct_feature(self, uid, query_rated_movie_ids,
                         query_rated_movie_scores, query_rated_genre_ids,
                         query_rated_genre_freqs, query_rated_genre_avg_scores,
                         candidate_movie_id, candidate_genre_ids,
                         ranking_candidate_movie_ids, label_rating_score):
    from preproc import movielens  # pylint: disable=g-import-not-at-top

    feature = {}
    feature[movielens.QUERY_USER_ID] = uid
    # Build candidate features.
    feature[movielens.CANDIDATE_MOVIE_ID] = candidate_movie_id
    feature[movielens.CANDIDATE_GENRE_IDS] = candidate_genre_ids
    # Define the label.
    feature[movielens.LABEL_RATING_SCORE] = label_rating_score
    # Build query features.
    feature[movielens.QUERY_RATED_MOVIE_IDS] = query_rated_movie_ids
    feature[movielens.QUERY_RATED_MOVIE_SCORES] = query_rated_movie_scores
    feature[movielens.QUERY_RATED_GENRE_IDS] = query_rated_genre_ids
    feature[movielens.QUERY_RATED_GENRE_FREQS] = query_rated_genre_freqs
    feature[movielens.QUERY_RATED_GENRE_AVG_SCORES] = (
        query_rated_genre_avg_scores)
    feature[movielens.RANKING_CANDIDATE_MOVIE_IDS] = ranking_candidate_movie_ids

    return feature

  def process(self, (user_id, user_ratings),
              movies_data,
              rating_threshold=3,
              is_ranking_problem=False,
              is_train=True,
              num_ranking_candidate_movie_ids=0,
              negative_sample_ratio=0,
              negative_sample_label=0.0,
              movie_rating_history=200):
    """Build feature columns for training examples.

    One example for each rating, which corresponds to a candidate, while the
    remaining ratings are used for constructing query features.

    Args:
      (user_id, user_ratings): id of the user, list of records containing all
        the needed data for a rating.
        Each record r is a dictionary. with r['movie'] containing features about
        the rated movie and r['rating'] containing fatures about the rating.
      movies_data: dict mapping movie id to the movie data.
      rating_threshold: only build examples for ratings above this threshold in
        evaluation, and set ratings lower than this score to 0 when eval_type
        is RANKING.
      is_ranking_problem: if set to True, discard ratings lower than
        rating_threshold in evaluation.
      is_train: if the processing is done for the training set.
      num_ranking_candidate_movie_ids: number of unrated movies to rank against
        the target movie in evaluation.
      negative_sample_ratio: number of negative samples to generate for
        each observed rating.
      negative_sample_label: the rating score to assign to a negative sample.
        Ignore this rating score when 'eval_type' is RANKING.
      movie_rating_history: Number of top ratings to keep for training,
        defaults to 200.
    Yields:
      Feature vectors, which are dictionaries from column names to values.
    """
    import collections  # pylint: disable=g-import-not-at-top
    from preproc import movielens  # pylint: disable=g-import-not-at-top

    # Convert to list because of_UnwindowedValues does not support len
    # remove once https://issues.apache.org/jira/browse/BEAM-1502 is fixed.
    def sorting_function(value):
      return (value['rating'], value['movie_id'])
    user_ratings = sorted(list(user_ratings), key=sorting_function,
                          reverse=True)
    input_size = len(user_ratings)
    movie_ids = []
    movie_ratings = []
    genre_freqs = collections.defaultdict(float)
    genre_ratings = collections.defaultdict(float)
    # Omit users who have too few or too many ratings.
    if input_size < 5:
      return
    if is_train:
      # For training keep only the top-k ratings.
      user_ratings = user_ratings[:movie_rating_history]
      input_size = len(user_ratings)
    else:
      # For evaluation discard users with more than 1000 ratings.
      if input_size > 1000:
        return
    for i, rating_data_i in enumerate(user_ratings):
      movie_id_i = rating_data_i['movie_id']
      rating_i = rating_data_i['rating']
      movie_data_i = movies_data[movie_id_i]
      genres_i = movie_data_i['genres']
      movie_ids.append(movie_id_i)
      movie_ratings.append(rating_i)
      for genre in genres_i:
        genre_freqs[genre] += 1.0
        genre_ratings[genre] += rating_i / (input_size - 1)

    # Use a leave one out based procedure to split user's data into the query
    # part and the candidate part.
    for i, rating_data_i in enumerate(user_ratings):
      movie_id_i = rating_data_i['movie_id']
      rating_i = rating_data_i['rating']
      if rating_i < rating_threshold and is_ranking_problem:
        # Skip low ratings when creating targets for evaluation data, and use
        # all the ratings as targets for training data.
        continue
      movie_data_i = movies_data[movie_id_i]
      genres_i = movie_data_i['genres']

      # Correct the genre frequency and avg rating due to leaving the
      # i-th rating out.
      for genre in genres_i:
        if int(genre_freqs[genre]) == 1:
          del genre_freqs[genre]
          del genre_ratings[genre]
        else:
          genre_freqs[genre] -= 1.0
          genre_ratings[genre] -= rating_i / (input_size - 1)

      # Add the genres to the features on a deterministic order.
      sorted_genre_freqs = collections.OrderedDict(
          sorted(genre_freqs.items()))
      sorted_genre_ratings = collections.OrderedDict(
          sorted(genre_ratings.items()))

      if num_ranking_candidate_movie_ids == 0:
        ranking_candidate_movie_ids = []
      else:
        ranking_candidate_movie_ids = movielens.create_random_movie_samples(
            movies_data.keys(), movie_ids, num_ranking_candidate_movie_ids,
            self._random_seed)

      feature = self._construct_feature(
          uid=user_id,
          query_rated_movie_ids=(movie_ids[:i] + movie_ids[i + 1:]),
          query_rated_movie_scores=(movie_ratings[:i] + movie_ratings[i + 1:]),
          query_rated_genre_ids=sorted_genre_freqs.keys(),
          query_rated_genre_freqs=sorted_genre_freqs.values(),
          query_rated_genre_avg_scores=(sorted_genre_ratings.values()),
          candidate_movie_id=[movie_id_i],
          candidate_genre_ids=genres_i,
          ranking_candidate_movie_ids=ranking_candidate_movie_ids,
          label_rating_score=rating_i)

      # Revert back corrections to genre frequency and avg ratings.
      for genre in genres_i:
        genre_freqs[genre] += 1.0
        genre_ratings[genre] += rating_i / (input_size - 1)

      yield feature

    # Generate negative samples from all movies except those rated by the user.
    if negative_sample_ratio <= 0:
      return
    negative_sample_size = int(input_size * negative_sample_ratio)
    negative_ids = movielens.create_random_movie_samples(
        movies_data.keys(), movie_ids, negative_sample_size, self._random_seed)
    sorted_genre_freqs = collections.OrderedDict(sorted(genre_freqs.items()))
    sorted_genre_ratings = collections.OrderedDict(
        sorted(genre_ratings.items()))
    for movie_id in negative_ids:
      movie_data = movies_data[movie_id]
      genres = movie_data['genres']
      feature = self._construct_feature(
          uid=user_id,
          query_rated_movie_ids=movie_ids,
          query_rated_movie_scores=movie_ratings,
          query_rated_genre_ids=sorted_genre_freqs.keys(),
          query_rated_genre_freqs=sorted_genre_freqs.values(),
          query_rated_genre_avg_scores=sorted_genre_ratings.values(),
          candidate_movie_id=[movie_id],
          candidate_genre_ids=genres,
          ranking_candidate_movie_ids=[],
          label_rating_score=negative_sample_label)
      yield feature


# TODO: Perhaps use Reshuffle (https://issues.apache.org/jira/browse/BEAM-1872)?
@beam.ptransform_fn
def _Shuffle(pcoll):  # pylint: disable=invalid-name
  """Shuffles a PCollection."""
  return (pcoll
          | 'PairWithRand' >> beam.Map(lambda x: (random.random(), x))
          | 'GroupByRand' >> beam.GroupByKey()
          | 'DropRand' >> beam.FlatMap(lambda (k, vs): vs))


def preprocess(pipeline, args):
  """Run pre-processing step as a pipeline.

  Args:
    pipeline: beam pipeline.
    args: parsed command line arguments.
  """
  from preproc import movielens  # pylint: disable=g-import-not-at-top

  # 1) Read the data into pcollections.
  movies_coder = tft_coders.CsvCoder(movielens.MOVIE_COLUMNS,
                                     movielens.make_movies_schema(),
                                     secondary_delimiter='|',
                                     multivalent_columns=['genres'])
  movies_data = (pipeline
                 | 'ReadMoviesData' >> beam.io.ReadFromText(
                     os.path.join(args.input_dir, 'movies.csv'),
                     coder=beam.coders.BytesCoder(),
                     # TODO(b/35653662): Obviate the need for setting this.
                     skip_header_lines=args.skip_header_lines)
                 | 'DecodeMovies' >> beam.Map(movies_coder.decode)
                 | 'KeyByMovie' >> beam.Map(lambda x: (x['movie_id'], x)))
  ratings_coder = tft_coders.CsvCoder(movielens.RATING_COLUMNS,
                                      movielens.make_ratings_schema())
  ratings_data = (pipeline
                  | 'ReadRatingsData' >> beam.io.ReadFromText(
                      os.path.join(args.input_dir, 'ratings*'),
                      skip_header_lines=args.skip_header_lines)
                  | 'DecodeRatings' >> beam.Map(ratings_coder.decode)
                  | 'KeyByUser' >> beam.Map(lambda x: (x['user_id'], x))
                  | 'GroupByUser' >> beam.GroupByKey())
  def train_eval_partition_fn((user_id, _), unused_num_partitions):
    return movielens.partition_fn(
        user_id, args.partition_random_seed, args.percent_eval)

  # Split train/eval data based on the integer user id.
  train_data, eval_data = (
      ratings_data
      | 'TrainEvalPartition'
      >> beam.Partition(train_eval_partition_fn, 2))

  movies_sideinput = beam.pvalue.AsDict(movies_data)
  train_data |= 'BuildTrainFeatures' >> beam.ParDo(
      BuildExampleFn(args.random_seed),
      movies_data=movies_sideinput,
      rating_threshold=0,
      is_ranking_problem=(args.eval_type == RANKING),
      is_train=True,
      num_ranking_candidate_movie_ids=0,
      negative_sample_ratio=args.negative_sample_ratio,
      negative_sample_label=args.negative_sample_label,
      movie_rating_history=args.movie_rating_history)

  movies_sideinput = beam.pvalue.AsDict(movies_data)
  eval_data |= 'BuildEvalFeatures' >> beam.ParDo(
      BuildExampleFn(args.random_seed),
      movies_data=movies_sideinput,
      rating_threshold=args.eval_score_threshold,
      is_ranking_problem=(args.eval_type == RANKING),
      is_train=False,
      num_ranking_candidate_movie_ids=args.num_ranking_candidate_movie_ids)

  # TFTransform based preprocessing.
  raw_metadata = dataset_metadata.DatasetMetadata(
      schema=movielens.make_examples_schema())
  _ = (raw_metadata
       | 'WriteRawMetadata' >> tft_beam_io.WriteMetadata(
           os.path.join(args.output_dir, 'raw_metadata'), pipeline))

  preprocessing_fn = movielens.make_preprocessing_fn()
  transform_fn = ((train_data, raw_metadata)
                  | 'Analyze' >> tft.AnalyzeDataset(preprocessing_fn))

  _ = (transform_fn
       | 'WriteTransformFn' >> tft_beam_io.WriteTransformFn(args.output_dir))

  @beam.ptransform_fn
  def TransformAndWrite(pcoll, path):  # pylint: disable=invalid-name
    pcoll |= 'Shuffle' >> _Shuffle()  # pylint: disable=no-value-for-parameter
    (dataset, metadata) = (((pcoll, raw_metadata), transform_fn)
                           | 'Transform' >> tft.TransformDataset())
    coder = tft_coders.ExampleProtoCoder(metadata.schema)
    _ = (dataset
         | 'SerializeExamples' >> beam.Map(coder.encode)
         | 'WriteExamples' >> beam.io.WriteToTFRecord(
             os.path.join(args.output_dir, path),
             file_name_suffix='.tfrecord.gz'))

  _ = train_data | 'TransformAndWriteTraining' >> TransformAndWrite(  # pylint: disable=no-value-for-parameter
      'features_train')

  _ = eval_data | 'TransformAndWriteEval' >> TransformAndWrite(  # pylint: disable=no-value-for-parameter
      'features_eval')

  # TODO(b/35300113) Remember to eventually also save the statistics.

  # Save files for online and batch prediction.
  prediction_schema = movielens.make_prediction_schema()
  prediction_coder = tft_coders.ExampleProtoCoder(prediction_schema)
  prediction_data = (
      eval_data
      | 'EncodePrediction' >> beam.Map(prediction_coder.encode))
  _ = (prediction_data
       | 'EncodePredictionAsB64Json' >> beam.Map(_encode_as_b64_json)
       | 'WritePredictDataAsText' >> beam.io.WriteToText(
           os.path.join(args.output_dir, 'features_predict'),
           file_name_suffix='.txt'))
  _ = (prediction_data
       | 'WritePredictDataAsTfRecord' >> beam.io.WriteToTFRecord(
           os.path.join(args.output_dir, 'features_predict'),
           file_name_suffix='.tfrecord.gz'))


def _encode_as_b64_json(serialized_example):
  import base64  # pylint: disable=g-import-not-at-top
  import json  # pylint: disable=g-import-not-at-top
  return json.dumps({'b64': base64.b64encode(serialized_example)})


def get_pipeline_name(runner, cloud):
  # Allow users to use cutom runner.
  if runner:
    return runner
  if cloud:
    return 'DataflowRunner'
  else:
    return 'DirectRunner'


def main(argv=None):
  """Run Preprocessing as a Dataflow."""

  args = parse_arguments(sys.argv if argv is None else argv)
  runner = get_pipeline_name(args.runner, args.cloud)
  if args.cloud:
    options = {
        'job_name': ('cloud-ml-sample-movielens-preprocess-{}'.format(
            datetime.datetime.now().strftime('%Y%m%d%H%M%S'))),
        'temp_location':
            os.path.join(args.output_dir, 'tmp'),
        'project':
            args.project_id,
        'max_num_workers':
            250,
        'setup_file':
            os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                'setup.py')),
    }
    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)
  else:
    pipeline_options = None

  temp_dir = os.path.join(args.output_dir, 'tmp')
  with beam.Pipeline(runner, options=pipeline_options) as pipeline:
    with tft.Context(temp_dir=temp_dir):
      preprocess(pipeline, args)


if __name__ == '__main__':
  main()
