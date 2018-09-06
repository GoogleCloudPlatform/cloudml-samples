import argparse
import os

import model

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def run_experiment(hparams):
  """Run the training and evaluate using the high level API"""
  train_input = model._make_training_input_fn(
      hparams.tft_working_dir,
      hparams.train_filebase,
      num_epochs=hparams.num_epochs,
      batch_size=hparams.train_batch_size,
      buffer_size=hparams.train_buffer_size,
      prefetch_buffer_size=hparams.train_prefetch_buffer_size)

  # Don't shuffle evaluation data
  eval_input = model._make_training_input_fn(
      hparams.tft_working_dir,
      hparams.eval_filebase,
      shuffle=False,
      batch_size=hparams.eval_batch_size,
      buffer_size=1,
      prefetch_buffer_size=hparams.eval_prefetch_buffer_size)

  train_spec = tf.estimator.TrainSpec(train_input,
                                      max_steps=hparams.train_steps
                                      )

  exporter = tf.estimator.FinalExporter('tft_classifier',
          model._make_serving_input_fn(hparams.tft_working_dir))

  eval_spec = tf.estimator.EvalSpec(eval_input,
                                    steps=hparams.eval_steps,
                                    exporters=[exporter],
                                    name='tft_classifier-eval'
                                    )

  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=hparams.job_dir)

  print('model dir {}'.format(run_config.model_dir))
  estimator = model.build_estimator(
      config=run_config,
      tft_working_dir=hparams.tft_working_dir,
      embedding_size=hparams.embedding_size,
      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(hparams.first_layer_size *
                     hparams.scale_factor**i))
          for i in range(hparams.num_layers)
      ],

  )

  tf.estimator.train_and_evaluate(estimator,
                                  train_spec,
                                  eval_spec)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--tft-working-dir',
      help='GCS or local paths to directory pointed by tf transform pipeline',
      required=True
  )
  parser.add_argument(
        '--train-filebase',
        help='Path to training data as in preprocessing.py',
        required=True
  )
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=256
  )
  parser.add_argument(
      '--train-buffer-size',
      help='Buffer size for the shuffle',
      type=int,
      default=None
  )
  parser.add_argument(
      '--train-prefetch-buffer-size',
      help='Number of example to prefetch',
      type=int,
      default=1
  )
  parser.add_argument(
        '--eval-filebase',
        help='Path to eval data as in preprocessing.py',
        required=True
  )
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for training steps',
      type=int,
      default=256
  )
  parser.add_argument(
      '--eval-prefetch-buffer-size',
      help='Number of example to prefetch',
      type=int,
      default=1
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
      default='INFO',
  )
  # Experiment arguments
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
      '--train-steps',
      help="""\
      Steps to run the training job for. Use for distributed training
      """,
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int
  )

  args = parser.parse_args()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  hparams=hparam.HParams(**args.__dict__)
  run_experiment(hparams)
