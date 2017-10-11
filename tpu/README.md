ResNet TensorFlow example to run on Cloud ML Engine.

The sample was branched from Cloud TPU Demos(
https://github.com/tensorflow/tpu-demos).

To train this model, follow the training quickstart [here](https://cloud.google.com/ml/docs/quickstarts/training).
After downloading and extracting the Cloud ML Engine samples, navigate to
the tpu sample directory:

```
cd cloudml-samples-master/tpu
```

And you can simply run the gcloud command to submit a TPU training job:

```
gcloud ml-engine jobs submit training $JOB_NAME \
  --config config.yaml \
  --job-dir $OUTPUT_PATH \
  --runtime-version HEAD \
  --module-name resnet.estimator_resnet \
  --package-path resnet/ \
  --region $REGION \
  -- \
  --data_dir=gs://cloudtpu-imagenet-data/train \
  --model_dir=$OUTPUT_PATH \
  --device=TPU \
  --use_fused_batchnorm=false \
  --batch_size=1024 \
  --model=resnet_v2_50 \
  --iterations_per_loop=50 \
  --input_layout=NHWC \
  --train_steps=10000 \
  --eval_steps=0 \
  --save_checkpoints_secs=0 \
  --map_threads=16 \
  --input_shuffle_capacity=10 \
  --dataset_reader_buffer_size=268435456
```
