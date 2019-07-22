# colab.research.google.com specific
if 'google.colab' in sys.modules:
    import json
    import os

    # TODO(user): change this
    args.model_dir = 'gs://your-gcs-bucket'

    # When connected to the TPU runtime
    if 'COLAB_TPU_ADDR' in os.environ:
        tpu_grpc = 'grpc://{}'.format(os.environ['COLAB_TPU_ADDR'])

        args.tpu = tpu_grpc
        args.use_tpu = True

        # Upload credentials to the TPU
        with tf.Session(tpu_grpc) as sess:
            data = json.load(open('/content/adc.json'))
            tf.contrib.cloud.configure_gcs(sess, credentials=data)
