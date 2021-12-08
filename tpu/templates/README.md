# TPU Templates

Templates for training TensorFlow models with Cloud TPUs.  They are minimal models with fake data that can be successfully trained on TPUs, and can be used as the starting point when you develop models from scratch.


 ... | estimator | keras | rewrite
 --- | --- | --- | ---
 cnn | [trainer.py](tpu_cnn_estimator/trainer.py)<br>[trainer.ipynb](tpu_cnn_estimator/trainer.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_cnn_estimator/trainer.ipynb)<br> |  | 
 dense | [trainer.py](tpu_estimator/trainer.py)<br>[trainer.ipynb](tpu_estimator/trainer.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_estimator/trainer.ipynb)<br> |  | [trainer_infeed_outfeed.py](tpu_rewrite/trainer_infeed_outfeed.py)<br>[trainer.py](tpu_rewrite/trainer.py)<br>[trainer_infeed_outfeed.ipynb](tpu_rewrite/trainer_infeed_outfeed.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_rewrite/trainer_infeed_outfeed.ipynb)<br>[trainer.ipynb](tpu_rewrite/trainer.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_rewrite/trainer.ipynb)<br>
 film | [trainer.py](tpu_film_estimator/trainer.py)<br>[trainer.ipynb](tpu_film_estimator/trainer.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_film_estimator/trainer.ipynb)<br> |  | 
 gan | [trainer_single.py](tpu_gan_estimator/trainer_single.py)<br>[trainer.py](tpu_gan_estimator/trainer.py)<br>[trainer_single.ipynb](tpu_gan_estimator/trainer_single.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_gan_estimator/trainer_single.ipynb)<br>[trainer.ipynb](tpu_gan_estimator/trainer.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_gan_estimator/trainer.ipynb)<br> |  | 
 grl | [trainer.py](tpu_grl_estimator/trainer.py)<br>[trainer.ipynb](tpu_grl_estimator/trainer.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_grl_estimator/trainer.ipynb)<br> |  | 
 lstm | [trainer.py](tpu_lstm_estimator/trainer.py)<br>[trainer.ipynb](tpu_lstm_estimator/trainer.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_lstm_estimator/trainer.ipynb)<br> | [trainer.py](tpu_lstm_keras/trainer.py)<br>[trainer.ipynb](tpu_lstm_keras/trainer.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_lstm_keras/trainer.ipynb)<br> | 
 triplet-loss | [trainer.py](tpu_triplet_loss_estimator/trainer.py)<br>[trainer.ipynb](tpu_triplet_loss_estimator/trainer.ipynb) [[Colab]](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_triplet_loss_estimator/trainer.ipynb)<br> |  | 

Note: The notebooks and the table above are generated with scripts in [tools](tools).

## TPUEstimator

Below we show how to run the basic [`tpu_estimator`](tpu_estimator) sample in three different ways to access TPUs: AI Platform, GCE, and Colab.

To run other samples replace `tpu_estimator` with their corresponding directory names.


### Train on AI Platform

Run from the `templates` directory.

```
BUCKET="gs://YOUR-GCS-BUCKET/"

TRAINER_PACKAGE_PATH="tpu_estimator"
MAIN_TRAINER_MODULE="tpu_estimator.trainer"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="tpu_estimator_$now"

JOB_DIR=$BUCKET"tpu_estimator/"$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-central1 \
    --config config.yaml \
    --runtime-version 1.9 \
    -- \
    --model-dir=$JOB_DIR\
    --use-tpu
```


### Train on Compute Engine

1. Create TPU from the console [TPU page](https://console.cloud.google.com/compute/tpus).  Record `TPU-NAME` of your choice.

1. Create a GCE VM in the same `ZONE` from the console [GCE VM page](https://console.cloud.google.com/compute/instances)

1. Connect to the GCE VM by clicking on the `SSH` button.

1. Run from the GCE VM:

    ```
    pip install tensorflow==1.9.0

    git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git

    cd cloudml-samples/tpu/templates

    python -m tpu_estimator.trainer \
    --use-tpu \
    --tpu TPU-NAME \
    --model-dir gs://YOUR-GCS-BUCKET/
    ```


### Train on Colab

1. Go to [Colab for the notebook](https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/tpu_estimator/trainer.ipynb).

1. Select TPU as the runtime type.

1. Update the GCS bucket for `model_dir` and run all cells.  You might be asked to authenticate in order to access the bucket.

