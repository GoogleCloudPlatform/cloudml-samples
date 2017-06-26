# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sample for building recommendation models for Movielens dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import json
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io

tf.logging.set_verbosity(tf.logging.INFO)

MODEL_TYPES = ['matrix_factorization', 'dnn_softmax']
MATRIX_FACTORIZATION, DNN_SOFTMAX = MODEL_TYPES

EVAL_TYPES = ['regression', 'ranking']
REGRESSION, RANKING = EVAL_TYPES

EMBEDDING_WEIGHT_INITIALIZERS = ['truncated_normal']
TRUNCATED_NORMAL = EMBEDDING_WEIGHT_INITIALIZERS

OPTIMIZERS = ['Adagrad', 'Adam', 'RMSProp']
ADAGRAD, ADAM, RMSPROP = OPTIMIZERS

MOVIE_VOCAB_SIZE = 28000
GENRE_VOCAB_SIZE = 20
# Use for initializing the global_rating_bias variable.
RATING_BIAS = 3.5

# Define a output alternative key for generating the top 100 candidates
# in exported model.
DEFAULT_OUTPUT_ALTERNATIVE = 'candidate_gen_100'

# Repeat the constants defined in movielens.py here as task.py will not be able
# to access movielens.py in SDK integration test.
"""Names of feature columns associated with the `Query`. These are the features
typically included in a recommendation request. In the case of movielens,
query contains just data about the user. In other applications, there
could be additional dimensions such as context (i.e. device, time of day, etc).
"""
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


"""Names of feature columns associated with the `Candidate`. These features
are used to match a candidate against the query."""
# The id of the candidate movie.
CANDIDATE_MOVIE_ID = 'cand_movie_id'
# The set of genres of the candidate movie.
CANDIDATE_GENRE_IDS = 'cand_genre_ids'
# The ranking candidate movie ids used to rank candidate movie against (used
# only in Eval graph).
RANKING_CANDIDATE_MOVIE_IDS = 'ranking_candidate_movie_ids'


"""Names of feature columns defining the label(s), which indicates how well
a candidate matches a query. There could be multiple labels in each instance.
Eg. We could have one label for the rating score and another label for the
number of times a user has watched the movie."""
LABEL_RATING_SCORE = 'label_rating_score'


def make_query_feature_columns():
  """Return feature columns associated with the query (i.e. user)."""
  query_movie_ids = tf.contrib.layers.sparse_column_with_integerized_feature(
      column_name=QUERY_RATED_MOVIE_IDS,
      bucket_size=MOVIE_VOCAB_SIZE)

  query_movie_ratings = tf.contrib.layers.weighted_sparse_column(
      query_movie_ids, QUERY_RATED_MOVIE_SCORES)

  query_genre_ids = tf.contrib.layers.sparse_column_with_integerized_feature(
      column_name=QUERY_RATED_GENRE_IDS,
      bucket_size=GENRE_VOCAB_SIZE)

  query_genre_weights = tf.contrib.layers.weighted_sparse_column(
      query_genre_ids, QUERY_RATED_GENRE_FREQS)

  query_genre_ratings = tf.contrib.layers.weighted_sparse_column(
      query_genre_ids, QUERY_RATED_GENRE_AVG_SCORES)

  return set([query_movie_ids, query_movie_ratings, query_genre_ids,
              query_genre_weights, query_genre_ratings])


def make_candidate_feature_columns():
  """Return feature columns associated with the candidate movie."""
  candidate_movie_id = (
      tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name=CANDIDATE_MOVIE_ID,
          bucket_size=MOVIE_VOCAB_SIZE))

  candidate_genre_ids = (
      tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name=CANDIDATE_GENRE_IDS,
          bucket_size=GENRE_VOCAB_SIZE))

  return set([candidate_movie_id, candidate_genre_ids])


def make_feature_columns():
  """Retrieve the feature columns required for training."""
  feature_columns = (make_query_feature_columns()
                     | make_candidate_feature_columns())
  # Add feature column for the label.
  target_rating_real_column = tf.contrib.layers.real_valued_column(
      column_name=LABEL_RATING_SCORE, dtype=tf.float32)
  feature_columns.add(target_rating_real_column)

  # Ranking candidate movies used only in eval graph to rank candidate movie
  # against.
  ranking_candidate_movie_ids = (
      tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name=RANKING_CANDIDATE_MOVIE_IDS,
          bucket_size=MOVIE_VOCAB_SIZE))
  feature_columns.add(ranking_candidate_movie_ids)

  return feature_columns


def make_input_fn(mode,
                  eval_type,
                  data_file_pattern,
                  randomize_input=None,
                  batch_size=None,
                  queue_capacity=None):
  """Provides input to the graph from file pattern.

  This function produces an input function that will feed data into
  the network. It will read the data from the files in data_file_pattern.

  Args:
    mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.
    eval_type: REGRESSION or RANKING.
    data_file_pattern: The file pattern to use to read in data. Required.
    randomize_input: Whether to randomize input.
    batch_size: The size of the batch when reading in data.
    queue_capacity: The queue capacity for the reader.

  Returns:
    A function that returns a dictionary of features and the target labels.
  """

  def _gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))

  def _input_fn():
    """Supplies input to our model.

    This function supplies input to our model, where this input is a
    function of the mode.

    Returns:
      A tuple consisting of 1) a dictionary of tensors whose keys are
      the feature names, and 2) a tensor of target labels if the mode
      is not INFER (and None, otherwise).
    Raises:
      ValueError: If data_file_pattern not set.
    """

    feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(
        feature_columns=make_feature_columns())
    if not data_file_pattern:
      raise ValueError('data_file_pattern must be set. Value provided: %s' %
                       data_file_pattern)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      num_epochs = None
    else:
      num_epochs = 1
    # TODO(nathanliu): remove this once TF 1.1 is out.
    file_pattern = (data_file_pattern[0] if len(data_file_pattern) == 1
                    else data_file_pattern)
    feature_map = tf.contrib.learn.io.read_batch_features(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        reader=_gzip_reader_fn,
        randomize_input=randomize_input,
        queue_capacity=queue_capacity,
        num_epochs=num_epochs)
    if eval_type == RANKING:
      # Ranking eval is based on precision/recall, which requires the targets
      # to be item ids.
      target = feature_map[CANDIDATE_MOVIE_ID]
    else:
      # We use RMSE/MAE for regression eval metrics, which require the targets
      # to be the actual rating scores.
      target = feature_map[LABEL_RATING_SCORE]
    return feature_map, target

  return _input_fn


def generate_top_k_scores_and_ids(logits, top_k):
  """This function computes top K ids and scores from logits tensor.

  Args:
    logits: logit tensor computed in the serving graph.
    top_k: number of top K elements to rank.

  Returns:
    predictions: scores of top K items.
    output_alternatives: ids of the top K items.
  """

  probabilities = tf.nn.softmax(
      logits, name=tf.contrib.learn.PredictionKey.PROBABILITIES)
  top_k_scores, top_k_ids = tf.nn.top_k(
      input=probabilities, k=top_k)
  top_k_ids = tf.contrib.lookup.index_to_string(
      tf.to_int64(top_k_ids),
      mapping=tf.constant([str(i) for i in xrange(MOVIE_VOCAB_SIZE)]))
  predictions = {
      # served as "scores" by Servo in the ClassificationResult
      tf.contrib.learn.PredictionKey.PROBABILITIES:
          top_k_scores,
      # served as "classes" by Servo in the ClassificationResult
      tf.contrib.learn.PredictionKey.CLASSES:
          top_k_ids
  }
  output_alternatives = {DEFAULT_OUTPUT_ALTERNATIVE: (
      tf.contrib.learn.ProblemType.CLASSIFICATION,
      predictions)}
  return predictions, output_alternatives


def model_builder(hparams):
  """Returns a function to build the model.

  Args:
    hparams: A named tuple with the following fields correspond to hyparameters.
      model_type - we support 2 types of models: MATRIX_FACTORIZATION and
        DNN_SOFTMAX.
      eval_type - REGRESSION or RANKING.
      learning_rate - learning rate for gradient based optimizers.
  Returns:
    A function to build the model's graph. This function is called by
    the Estimator object to construct the graph.
  Raises:
    ValueError: When the wrong evaluation type is specified.
  """

  def _matrix_factorization_model_fn(features, target_ratings, mode):
    """Creates a neighborhood factorization model.

    Each user is represented by a combination of embeddings of rated items,
    as described in the paper: "Factorization Meets the Neighborhood:
    a Multifaceted Collaborative Filtering Model - Yehuda Koren (KDD 2013)".

    Args:
      features: A dictionary of tensors keyed by the feature name.
      target_ratings: A tensor representing the labels (in this case,
        the ratings on the target movie).
      mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.

    Returns:
      ModelFnOps with the mode, prediction, loss, train_op and
      output_alternatives a dictionary specifying the output for a
      classification request during serving.
    """
    _ = target_ratings  # Unused on this model.
    if hparams.embedding_weight_initializer == TRUNCATED_NORMAL:
      embedding_weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
    else:
      embedding_weight_initializer = None
    query_movie_embedding_weights = tf.get_variable(
        'query_movie_ids_embedding_weights',
        [MOVIE_VOCAB_SIZE, hparams.movie_embedding_dim],
        initializer=embedding_weight_initializer,
        regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_weight_decay))
    query_movie_ids = features[QUERY_RATED_MOVIE_IDS]
    query_embeddings = tf.nn.embedding_lookup_sparse(
        [query_movie_embedding_weights],
        query_movie_ids,
        None,
        combiner='sqrtn',
        name='query_embedding')
    query_biases, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        columns_to_tensors=features,
        feature_columns=make_query_feature_columns(),
        num_outputs=1)
    global_rating_bias = tf.get_variable(
        name='global_rating_bias',
        initializer=tf.constant(RATING_BIAS, dtype=tf.float32))
    candidate_movie_embedding_weights = tf.get_variable(
        'candidate_movie_id_embedding_weights',
        [MOVIE_VOCAB_SIZE, hparams.movie_embedding_dim],
        initializer=embedding_weight_initializer,
        regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_weight_decay))
    candidate_biases, _, _ = (
        tf.contrib.layers.weighted_sum_from_feature_columns(
            columns_to_tensors=features,
            feature_columns=make_candidate_feature_columns(),
            num_outputs=1))

    # Create layers for target features.
    if mode != tf.contrib.learn.ModeKeys.INFER:
      candidate_movie_ids = features[CANDIDATE_MOVIE_ID]
      candidate_embeddings = tf.nn.embedding_lookup_sparse(
          [candidate_movie_embedding_weights],
          candidate_movie_ids,
          None,
          name='candidate_embedding')
      predictions = tf.reduce_sum(tf.multiply(
          query_embeddings, candidate_embeddings), 1, keep_dims=True)
      if hparams.enable_bias:
        biases = tf.add(query_biases, candidate_biases)
        predictions = tf.add(predictions, biases)
        predictions = tf.add(predictions, global_rating_bias)

      labels = features[LABEL_RATING_SCORE]
      loss = tf.losses.mean_squared_error(labels, predictions)

      if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=hparams.learning_rate,
            optimizer=hparams.optimizer)
        return tf.contrib.learn.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)
      elif mode == tf.contrib.learn.ModeKeys.EVAL:
        if hparams.eval_type == REGRESSION:
          return tf.contrib.learn.ModelFnOps(
              mode=mode, predictions=predictions, loss=loss)
        elif hparams.eval_type == RANKING:
          # For 'RANKING' eval, we are interested in precision@k, recall@k
          # metrics which require us to compute prediction/ranking scores for
          # all movies.
          predictions = tf.matmul(query_embeddings,
                                  candidate_movie_embedding_weights,
                                  transpose_b=True)
          if hparams.enable_bias:
            biases = tf.add(query_biases, candidate_biases)
            predictions = tf.add(predictions, biases)

          if hparams.use_ranking_candidate_movie_ids:
            # Get ranking candidate movie ids to rank our candidate movie
            # against.
            ranking_candidate_movie_ids = features[RANKING_CANDIDATE_MOVIE_IDS]
            movies_to_rank_condition = tf.sparse_to_indicator(
                tf.sparse_concat(
                    axis=1,
                    sp_inputs=[ranking_candidate_movie_ids,
                               candidate_movie_ids]),
                MOVIE_VOCAB_SIZE)
            predictions = tf.where(movies_to_rank_condition, predictions,
                                   tf.fill(
                                       tf.shape(predictions),
                                       tf.reduce_min(predictions)))
          return tf.contrib.learn.ModelFnOps(
              mode=mode,
              predictions=predictions,
              loss=loss)
    elif mode == tf.contrib.learn.ModeKeys.INFER:
      scores = tf.matmul(query_embeddings,
                         candidate_movie_embedding_weights,
                         transpose_b=True)
      if hparams.enable_bias:
        biases = tf.add(query_biases, candidate_biases)
        scores = tf.add(scores, biases)

      # Eliminate already rated candates.
      rated_movie_ids = features[QUERY_RATED_MOVIE_IDS]
      pruned_scores = tf.where(
          tf.sparse_to_indicator(rated_movie_ids, MOVIE_VOCAB_SIZE),
          tf.fill(tf.shape(scores), tf.reduce_min(scores)), scores)
      predictions, output_alternatives = generate_top_k_scores_and_ids(
          pruned_scores, hparams.top_k_infer)

      return tf.contrib.learn.ModelFnOps(
          mode=mode,
          predictions=predictions,
          output_alternatives=output_alternatives)

  def _embed_query_features(features, mode):
    """Build a DNN that produces dense embeddings of queries."""
    if hparams.embedding_weight_initializer == TRUNCATED_NORMAL:
      embedding_weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
    else:
      embedding_weight_initializer = None
    movie_ids_embedding_weights = tf.get_variable(
        'query_movie_ids_embedding_weights',
        [MOVIE_VOCAB_SIZE, hparams.movie_embedding_dim],
        initializer=embedding_weight_initializer,
        regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_weight_decay))
    movie_ids = features[QUERY_RATED_MOVIE_IDS]
    movie_ratings = features[QUERY_RATED_MOVIE_SCORES]
    movies_embedding = tf.contrib.layers.safe_embedding_lookup_sparse(
        [movie_ids_embedding_weights],
        movie_ids,
        movie_ratings,
        combiner='sqrtn',
        name='query_movies_embedding')

    genres_embedding_weights = tf.get_variable(
        'query_genres_embedding_weights',
        [GENRE_VOCAB_SIZE, hparams.genre_embedding_dim],
        initializer=embedding_weight_initializer,
        regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_weight_decay))
    genre_ids = features[QUERY_RATED_GENRE_IDS]
    genre_freqs = features[QUERY_RATED_GENRE_FREQS]
    genre_ratings = features[QUERY_RATED_GENRE_AVG_SCORES]
    genres_embedding_freqs = tf.contrib.layers.safe_embedding_lookup_sparse(
        [genres_embedding_weights],
        genre_ids,
        genre_freqs,
        combiner='sqrtn',
        name='query_genres_embedding_freqs')
    genres_embedding_ratings = tf.contrib.layers.safe_embedding_lookup_sparse(
        [genres_embedding_weights],
        genre_ids,
        genre_ratings,
        combiner='sqrtn',
        name='query_genres_embedding_ratings')

    bottom_layer = tf.concat(
        [movies_embedding, genres_embedding_freqs, genres_embedding_ratings], 1,
        name='query_bottom_layer')

    if hparams.enable_batch_norm:
      normalizer_fn = tf.contrib.layers.batch_norm
      normalizer_params = {'is_training':
                           mode == tf.contrib.learn.ModeKeys.TRAIN}
    else:
      normalizer_fn = None
      normalizer_params = None
    return tf.contrib.layers.stack(
        inputs=bottom_layer,
        layer=tf.contrib.layers.fully_connected,
        stack_args=hparams.query_hidden_dims,
        weights_regularizer=tf.contrib.layers.l2_regularizer(
            hparams.l2_weight_decay),
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params)

  def _dnn_softmax_fn(features, targets, mode):
    """Creates the prediction, loss, and train ops.

    Args:
      features: A dictionary of tensors keyed by the feature name.
      targets: A tensor representing the labels (in this case,
        the ratings on the target movie).
      mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.

    Returns:
      ModelFnOps with the mode, prediction, loss, train_op and
      output_alternatives a dictionary specifying the output for a
      classification request during serving.
    Raises:
      ValueError: When the wrong evaluation type is specified.
    """
    _ = targets  # Unused variable.
    class_weights = tf.get_variable(
        name='class_weights',
        shape=[MOVIE_VOCAB_SIZE, hparams.query_hidden_dims[-1]],
        initializer=tf.contrib.layers.xavier_initializer())
    class_biases = tf.get_variable(
        name='class_biases',
        shape=[MOVIE_VOCAB_SIZE],
        initializer=tf.zeros_initializer())
    query_embeddings = _embed_query_features(features, mode=mode)
    tf.summary.scalar('query_embeddings_zero_fraction',
                      tf.nn.zero_fraction(query_embeddings))

    # Create layers for target features.
    if mode != tf.contrib.learn.ModeKeys.INFER:
      logits_layer = tf.matmul(
          query_embeddings, tf.transpose(class_weights)) + class_biases
      target_one_hot = tf.one_hot(
          indices=features[CANDIDATE_MOVIE_ID].values,
          depth=MOVIE_VOCAB_SIZE,
          on_value=1.0)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          labels=target_one_hot, logits=logits_layer))

      if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=hparams.learning_rate,
            optimizer=hparams.optimizer)
        return tf.contrib.learn.ModelFnOps(
            mode=mode, loss=loss, train_op=train_op)
      elif mode == tf.contrib.learn.ModeKeys.EVAL:
        if hparams.eval_type == REGRESSION:
          raise ValueError('eval_type must be RANKING for DNN softmax model.')
        elif hparams.eval_type == RANKING:
          predictions = tf.matmul(
              query_embeddings, tf.transpose(class_weights)) + class_biases
          if hparams.use_ranking_candidate_movie_ids:
            # Get ranking candidate movie ids to rank our candidate movie
            # against.
            ranking_candidate_movie_ids = features[RANKING_CANDIDATE_MOVIE_IDS]
            movies_to_rank_condition = tf.sparse_to_indicator(
                tf.sparse_concat(
                    axis=1,
                    sp_inputs=[ranking_candidate_movie_ids,
                               features[CANDIDATE_MOVIE_ID]]),
                MOVIE_VOCAB_SIZE)
            predictions = tf.where(movies_to_rank_condition, predictions,
                                   tf.fill(
                                       tf.shape(predictions),
                                       tf.reduce_min(predictions)))
          return tf.contrib.learn.ModelFnOps(
              mode=mode, predictions=predictions, loss=loss)
    elif mode == tf.contrib.learn.ModeKeys.INFER:
      scores = tf.matmul(
          query_embeddings, tf.transpose(class_weights)) + class_biases

      rated_movie_ids = features[QUERY_RATED_MOVIE_IDS]
      pruned_scores = tf.where(
          tf.sparse_to_indicator(rated_movie_ids, MOVIE_VOCAB_SIZE),
          tf.fill(tf.shape(scores), tf.reduce_min(scores)), scores)
      predictions, output_alternatives = generate_top_k_scores_and_ids(
          pruned_scores, hparams.top_k_infer)
      return tf.contrib.learn.ModelFnOps(
          mode=mode,
          predictions=predictions,
          output_alternatives=output_alternatives)

  if hparams.model_type == MATRIX_FACTORIZATION:
    return _matrix_factorization_model_fn
  elif hparams.model_type == DNN_SOFTMAX:
    return _dnn_softmax_fn
  else:
    raise ValueError


def create_evaluation_metrics(eval_type):
  """Creates the evaluation metrics for the model.

  Args:
    eval_type: one of EVAL_TYPES.

  Returns:
    A dictionary with keys that are strings naming the evaluation
    metrics and values that are functions taking arguments of
    (predictions, targets), returning a tuple of a tensor of the
    metric's value together with an op to update the metric's value.
  """
  if eval_type == REGRESSION:
    return {
        'mae': (
            tf.contrib.learn.MetricSpec(
                metric_fn=functools.partial(
                    tf.contrib.metrics.streaming_mean_absolute_error))),
        'rmse': (
            tf.contrib.learn.MetricSpec(
                metric_fn=functools.partial(
                    tf.contrib.metrics.streaming_root_mean_squared_error)))
    }
  else:  # RANKING
    eval_metrics = {}
    for k in [1, 5, 10, 20, 35, 50, 75, 100]:
      eval_metrics['precision_at_%02d' % k] = (
          tf.contrib.learn.MetricSpec(
              metric_fn=functools.partial(
                  tf.contrib.metrics.streaming_sparse_precision_at_k, k=k)))
      eval_metrics['recall_at_%02d' % k] = (
          tf.contrib.learn.MetricSpec(
              metric_fn=functools.partial(
                  tf.contrib.metrics.streaming_sparse_recall_at_k, k=k)))
    for k in [1, 2, 3, 4, 5, 10, 20]:
      eval_metrics['mean_average_precision_at_%d' % k] = (
          tf.contrib.learn.MetricSpec(
              metric_fn=functools.partial(
                  tf.contrib.metrics.streaming_sparse_average_precision_at_k,
                  k=k)))
    return eval_metrics


def make_experiment_fn(args):
  """Wrap the get experiment function to provide the runtime arguments."""

  def make_experiment(output_dir):
    """Function that creates an experiment http://goo.gl/HcKHlT.

    Args:
      output_dir: The directory where the training output should be written.
    Returns:
      A `tf.contrib.learn.Experiment`.
    """

    estimator = tf.contrib.learn.Estimator(
        model_fn=model_builder(hparams=args),
        model_dir=output_dir)

    train_input_fn = make_input_fn(
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        eval_type=args.eval_type,
        data_file_pattern=args.train_data_paths,
        randomize_input=args.randomize_input,
        batch_size=args.batch_size,
        queue_capacity=4 * args.batch_size)

    eval_input_fn = make_input_fn(
        mode=tf.contrib.learn.ModeKeys.EVAL,
        eval_type=args.eval_type,
        data_file_pattern=args.eval_data_paths,
        batch_size=args.eval_batch_size,
        queue_capacity=4 * args.eval_batch_size)

    raw_metadata = metadata_io.read_metadata(args.raw_metadata_path)
    # Both ratings and candidate features are not needed for serving.
    raw_label_keys = [LABEL_RATING_SCORE]
    # For serving, we only need query features.
    raw_feature_keys = [QUERY_RATED_MOVIE_IDS,
                        QUERY_RATED_MOVIE_SCORES,
                        QUERY_RATED_GENRE_IDS,
                        QUERY_RATED_GENRE_FREQS,
                        QUERY_RATED_GENRE_AVG_SCORES]
    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(
            raw_metadata,
            args.transform_savedmodel,
            raw_label_keys=raw_label_keys,
            raw_feature_keys=raw_feature_keys))

    export_strategy = tf.contrib.learn.utils.make_export_strategy(
        serving_input_fn,
        default_output_alternative_key=DEFAULT_OUTPUT_ALTERNATIVE)

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_steps=(args.train_steps or
                     args.num_epochs * args.train_set_size // args.batch_size),
        eval_steps=args.eval_steps,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        eval_metrics=create_evaluation_metrics(args.eval_type),
        export_strategies=[export_strategy],
        # Do not remove this is needed until b/36498507 is fixed.
        min_eval_frequency=1000)

  # Return a function to create an Experiment.
  return make_experiment


def create_parser():
  """Initialize command line parser using argparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_type',
      help='Model type to train on',
      choices=MODEL_TYPES,
      default=MATRIX_FACTORIZATION)
  parser.add_argument(
      '--eval_type',
      help='Evaluation type',
      choices=EVAL_TYPES,
      default=REGRESSION)
  parser.add_argument(
      '--train_data_paths', type=str, action='append', required=True)
  parser.add_argument(
      '--eval_data_paths', type=str, action='append', required=True)
  parser.add_argument('--output_path', type=str, required=True)
  # The following two parameters are required for tf.Transform.
  parser.add_argument('--raw_metadata_path', type=str, required=True)
  parser.add_argument('--transform_savedmodel', type=str, required=True)
  parser.add_argument(
      '--query_hidden_dims',
      nargs='*',
      help='List of hidden units per layer. All layers are fully connected. Ex.'
      '`128 64` means first layer has 128 nodes and second one has 64.',
      default=[64, 32],
      type=int)
  parser.add_argument(
      '--candidate_hidden_dims',
      nargs='*',
      help='List of hidden units per layer. All layers are fully connected. Ex.'
      '`128 64` means first layer has 128 nodes and second one has 64.',
      default=[64, 32],
      type=int)
  parser.add_argument(
      '--batch_size',
      help='Number of input records used per batch.',
      default=128,
      type=int)
  parser.add_argument(
      '--randomize_input',
      action='store_true',
      default=True,
      help='Whether to randomize inputs data.')
  parser.add_argument(
      '--eval_batch_size',
      help='Number of eval records used per batch.',
      default=128,
      type=int)
  parser.add_argument(
      '--learning_rate',
      help='Learning rate',
      default=0.01,
      type=float)
  parser.add_argument(
      '--l2_weight_decay',
      help='L2 regularization strength',
      default=0.001,
      type=float)
  parser.add_argument(
      '--movie_embedding_dim',
      help='Dimensionality of movie embeddings.',
      default=64,
      type=int)
  parser.add_argument(
      '--genre_embedding_dim',
      default=8,
      help='Dimensionality of genre embeddings.',
      type=int)
  parser.add_argument(
      '--enable_bias',
      action='store_true',
      default=False,
      help='Whether to learn per user/item bias.')
  parser.add_argument(
      '--train_steps', help='Number of training steps to perform.', type=int)
  parser.add_argument(
      '--eval_steps',
      help='Number of evaluation steps to perform.',
      type=int,
      default=500)
  parser.add_argument(
      '--train_set_size',
      help='Number of samples on the train dataset.',
      type=int)
  parser.add_argument(
      '--num_epochs', help='Number of epochs', default=5, type=int)
  parser.add_argument(
      '--top_k_infer',
      help='Number of candidates to return during inference stage',
      default=100,
      type=int)
  parser.add_argument(
      '--embedding_weight_initializer',
      help='Embedding weight initializer',
      choices=EMBEDDING_WEIGHT_INITIALIZERS)
  parser.add_argument(
      '--optimizer',
      help='Optimizer to use',
      choices=OPTIMIZERS,
      default=ADAGRAD)
  parser.add_argument(
      '--enable_batch_norm',
      action='store_true',
      default=False,
      help='Whether to use batch normalization in DNN model.')
  parser.add_argument(
      '--use_ranking_candidate_movie_ids',
      action='store_true',
      default=False,
      help='Whether to use ranking candidate movies to rank our target movie'
      'against.')
  return parser


def main(argv=None):
  """Run a Tensorflow model on the Movielens dataset."""
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task') or {'type': 'master', 'index': 0}
  argv = sys.argv if argv is None else argv
  args = create_parser().parse_args(args=argv[1:])

  trial = task_data.get('trial')
  if trial is not None:
    output_path = os.path.join(args.output_path, trial)
  else:
    output_path = args.output_path

  learn_runner.run(experiment_fn=make_experiment_fn(args),
                   output_dir=output_path)


if __name__ == '__main__':
  main()
