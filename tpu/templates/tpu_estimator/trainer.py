import argparse
import numpy as np
import tensorflow as tf


def model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()
    hidden = tf.layers.dense(features, 10, activation=tf.nn.relu)
    output = tf.layers.dense(hidden, 1)

    predictions = output
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        loss = tf.nn.l2_loss(predictions - labels)

        # define train_op
        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)

        # wrapper to make the optimizer work with TPUs
        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=global_step)

    if params['use_tpu']:
        # TPU version of EstimatorSpec
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op
        )
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op
        )


def train_input_fn(params={}):
    # make some fake data
    x = np.random.rand(100, 5)
    w = np.random.rand(5)
    y = np.sum(x * w, axis=1)

    # TPUs currently do not support float64
    x_tensor = tf.constant(x, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.float32)

    # create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

    # TPUEstimator passes params when calling input_fn
    batch_size = params.get('batch_size', 16)

    dataset = dataset.repeat().shuffle(32).batch(batch_size)

    # TPUs need to know all dimensions when the graph is built
    # Datasets know the batch size only when the graph is run
    def set_shapes(features, labels):
        features_shape = features.get_shape().merge_with([batch_size, None])
        labels_shape = labels.get_shape().merge_with([batch_size])

        features.set_shape(features_shape)
        labels.set_shape(labels_shape)

        return features, labels

    dataset = dataset.map(set_shapes)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def main(args):
    # pass the args as params so the model_fn can use
    # the TPU specific args
    params = vars(args)

    if args.use_tpu:
        # additional configs required for using TPUs
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu)
        tpu_config = tf.contrib.tpu.TPUConfig(
            num_shards=8 # using Cloud TPU v2-8
        )

        # use the TPU version of RunConfig
        config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=args.model_dir,
            tpu_config=tpu_config,
            save_checkpoints_steps=100,
            save_summary_steps=100
        )

        # TPUEstimator
        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            config=config,
            params=params,
            train_batch_size=32, # FIXME
            eval_batch_size=32, # FIXME
            export_to_tpu=False
        )
    else:
        config = tf.estimator.RunConfig(model_dir=args.model_dir)

        estimator = tf.estimator.Estimator(
            model_fn,
            config=config,
            params=params
        )

    estimator.train(train_input_fn, max_steps=args.max_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tpu-template'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--use-tpu',
        action='store_true'
    )
    parser.add_argument(
        '--tpu',
        default=None
    )

    args, _ = parser.parse_known_args()

    main(args)
