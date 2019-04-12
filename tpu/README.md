Tensor Processing Units (TPUs) are Google’s custom-developed ASICs used to accelerate machine-learning workloads. You can run your training jobs on AI Platform, using Cloud TPU. AI Platform provides a job management interface so that you don't need to manage the TPU yourself. Instead, you can use the AI Platform jobs API in the same way as you use it for training on a CPU or a GPU.

For TPU training, we recommend users to use the sample from [Cloud TPU Demos](https://github.com/tensorflow/tpu). And you can follow the [doc](https://cloud.google.com/ml-engine/docs/tensorflow/using-tpus) to get started with TPU training on AI Platform.

## Samples:
* [Tensorflow: ResNet](training/resnet) - Using the ResNet-50 dataset with Cloud TPUs on AI Platform.
* [Tensorflow: HP Tuning - ResNet](hptuning/resent-hptuning) - How to run hyperparameter tuning jobs on AI Platform with Cloud TPUs using TensorFlow's tf.metrics.
* [Tensorflow: Hypertune - ResNet](hptuning/resent-hypertune) - How to run hyperparameter tuning jobs on AI Platform with Cloud TPUs using the cloudml-hypertune package.
* [Tensorflow: Templates](templates) - A collection of minimal templates that can be run on Cloud TPUs on Compute Engine, Cloud Machine Learning, and Colab.

If you’re looking for samples for how to use Cloud TPU, check out the guides here. 

Note: These guides do not use AI Platform
* [MNIST on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/mnist)
* [ResNet-50 on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/resnet)
* [Inception on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/inception)
* [Advanced Inception v3 on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/inception-v3-advanced)
* [RetinaNet on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/retinanet)
* [Transformer with Tensor2Tensor on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/transformer)
