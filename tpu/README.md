If you want to try TPU on Cloud ML Engine, please contact
`cloudml-feedback@google.com`

For TPU training, we recommend users to use the sample from [Cloud TPU Demos](
https://github.com/tensorflow/tpu-demos).


Take the resnet garden model as example, to train this model,
follow the training quickstart [here](https://cloud.google.com/ml/docs/quickstarts/training).
The only difference is that you should download the sample zip file from
[Cloud TPU github repository](
https://github.com/tensorflow/tpu-demos).

After downloading and extracting the Cloud TPU samples, navigate to
the tpu sample directory:

```
cd tpu-demos-master/cloud_tpu/models/
```

Then, create the config file:

```
cat > config.yaml << EOF
trainingInput:
  scaleTier: CUSTOM
  masterType: standard
  workerType: standard_tpu
  workerCount: 1
EOF
```

And you can simply run the gcloud command to submit a TPU training job:

```
gcloud ml-engine jobs submit training $JOB_NAME \
  --config config.yaml \
  --job-dir $OUTPUT_PATH \
  --runtime-version HEAD \
  --module-name resnet_garden.resnet_main \
  --package-path resnet_garden/ \
  --region $REGION \
  -- \
  --data_dir=gs://cloudtpu-imagenet-data/train \
  --model_dir=$OUTPUT_PATH \
  --train_steps=5000

```
