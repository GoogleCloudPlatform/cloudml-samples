import argparse

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)


def generate_experiment_fn(train_file,
                           eval_file,
                           num_epochs=None,
                           train_batch_size=40,
                           eval_batch_size=40,
                           embedding_size=8,
                           first_layer_size=100,
                           num_layers=4,
                           scale_factor=0.7,
                           **experiment_args):
  def _experiment_fn(output_dir):
    train_input = model.generate_input_fn(
        train_file,
        num_epochs=num_epochs,
        batch_size=train_batch_size,
        skip_header_lines=model.TRAIN_HEADER_LINES
    )
    eval_input = model.generate_input_fn(
        eval_file,
        batch_size=eval_batch_size,
        skip_header_lines=model.EVAL_HEADER_LINES,
        shuffle=False
    )
    return tf.contrib.learn.Experiment(
        model.build_estimator(
            job_dir,
            embedding_size=embedding_size,
            hidden_units=[
                int(first_layer_size * scale_factor**i)
                for i in range(num_layers)
            ]
        ),
        train_input_fn=train_input,
        eval_input_fn=eval_input,
        export_strategies=[saved_model_export_utils.make_export_strategy(
            model.serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1
        )],
        **experiment_args
    )
  return _experiment_fn


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--train-file',
      help='GCS or local path to training data',
      required=True
  )
  parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      type=int,
  )
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=40
  )
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=40
  )
  parser.add_argument(
      '--train-steps',
      help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int
  )
  parser.add_argument(
      '--eval-file',
      help='GCS or local path to evaluation data',
      required=True
  )
  # Training arguments
  parser.add_argument(
      '--embedding-size',
      help='Number of embedding dimensions for categorical columns',
      default=8,
      type=int
  )
  parser.add_argument(
      '--first-layer-size',
      help='Number of nodes in the first layer of the DNN',
      default=100,
      type=int
  )
  parser.add_argument(
      '--num-layers',
      help='Number of layers in the DNN',
      default=4,
      type=int
  )
  parser.add_argument(
      '--scale-factor',
      help='How quickly should the size of the layers in the DNN decay',
      default=0.7,
      type=float
  )
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )

  # Experiment arguments
  parser.add_argument(
      '--eval-delay-secs',
      help='How long to wait before running first evaluation',
      default=10,
      type=int
  )
  parser.add_argument(
      '--min-eval-frequency',
      help='Minimum number of training steps between evaluations',
      default=1,
      type=int
  )

  args = parser.parse_args()
  arguments = args.__dict__
  job_dir = arguments.pop('job_dir')

  # Run the training job
  learn_runner.run(generate_experiment_fn(**arguments), job_dir)
