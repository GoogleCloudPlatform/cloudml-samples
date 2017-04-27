import argparse

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)


def generate_experiment_fn(train_files,
                           eval_files,
                           num_epochs=None,
                           train_batch_size=40,
                           eval_batch_size=40,
                           embedding_size=8,
                           first_layer_size=100,
                           num_layers=4,
                           scale_factor=0.7,
                           **experiment_args):
  """Create an experiment function given hyperparameters.

  See command line help text for description of args.
  Returns:
    A function (output_dir) -> Experiment where output_dir is a string
    representing the location of summaries, checkpoints, and exports.
    this function is used by learn_runner to create an Experiment which
    executes model code provided in the form of an Estimator and
    input functions.

    All listed arguments in the outer function are used to create an
    Estimator, and input functions (training, evaluation, serving).
    Unlisted args are passed through to Experiment.
  """
  def _experiment_fn(output_dir):
    # num_epochs can control duration if train_steps isn't
    # passed to Experiment
    train_input = model.generate_input_fn(
        train_files,
        num_epochs=num_epochs,
        batch_size=train_batch_size,
    )
    # Don't shuffle evaluation data
    eval_input = model.generate_input_fn(
        eval_files,
        batch_size=eval_batch_size,
        shuffle=False
    )
    return tf.contrib.learn.Experiment(
        model.build_estimator(
            output_dir,
            embedding_size=embedding_size,
            # Construct layers sizes with exponetial decay
            hidden_units=[
                max(2, int(first_layer_size * scale_factor**i))
                for i in range(num_layers)
            ]
        ),
        train_input_fn=train_input,
        eval_input_fn=eval_input,
        # export strategies control the prediction graph structure
        # of exported binaries.
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
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
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
      '--eval-files',
      help='GCS or local paths to evaluation data',
      nargs='+',
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

  # Argument to turn on all logging
  parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default=tf.logging.FATAL,
      help='Set logging verbosity'
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
  tf.logging.set_verbosity(arguments.pop('verbosity'))

  job_dir = arguments.pop('job_dir')

  print('Starting Census: Please lauch tensorboard to see results:\n'
        'tensorboard --logdir=$MODEL_DIR')

  # Run the training job
  # learn_runner pulls configuration information from environment
  # variables using tf.learn.RunConfig and uses this configuration
  # to conditionally execute Experiment, or param server code
  learn_runner.run(generate_experiment_fn(**arguments), job_dir)
