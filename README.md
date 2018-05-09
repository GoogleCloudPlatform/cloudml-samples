# Cloud Machine Learning Engine

This repository contains samples for usage of the Google Cloud Machine Learning Engine (Cloud ML Engine). For installation instructions and overview, please see [the documentation](https://cloud.google.com/ml-engine/docs/). Please refer to **README.md** in each sample directory for more specific instructions.

- - -
## ML on GCP guides
Please checkout [ML on GCP](https://github.com/GoogleCloudPlatform/ml-on-gcp) guides on how to bring your code from various ML frameworks to [Google Cloud Platform](https://cloud.google.com/).

## Cloud ML Engine

### CPU and GPU

* [TensorFlow Estimator Trainer Package Template](cloudml-template) -
  Simplifies creating a trainer package for Cloud ML Engine.
* [Census](census) - Predict a person's income level
  * [Keras Census](census/keras) - [Keras](https://keras.io/) single node model
  * [Canned Estimator](census/estimator) - [TensorFlow canned estimator](https://www.tensorflow.org/programmers_guide/estimators#pre-made_estimators) model
  * [Custom Estimator](census/customestimator) - [TensorFlow custom estimator](https://www.tensorflow.org/programmers_guide/estimators#custom_estimators) model
  * [Low Level TF](census/tensorflowcore) - TensorFlow low level API model
  * [scikit-learn Online Prediction](sklearn/notebooks/Online%20Prediction%20with%20scikit-learn.ipynb)
  * [XGBoost Online Prediction](xgboost/notebooks/Online%20Prediction%20with%20XGBoost.ipynb)
* [Criteo](criteo_tft) - Predict how likely a person is to click on an
  advertisement
* [Flowers](flowers) - Recognize the type of a flower from its image
* [Movielens](movielens) - Make movie recommendations
* [Reddit](reddit_tft) - Predict the score of a Reddit thread using a wide and deep model
* [Cifar10](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) - Classify image content (training on multiple GPUs)

## Cloud TPU
* [MNIST on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/mnist)
* [ResNet-50 on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/resnet)
* [Inception on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/inception)
* [Advanced Inception v3 on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/inception-v3-advanced)
* [RetinaNet on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/retinanet)
* [Transformer with Tensor2Tensor on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/transformer)

- - -

## Google Samples

* [Genomics Ancestry Inference](https://github.com/googlegenomics/cloudml-examples) - Genomics ancestry inference using 1000 Genomes dataset

- - -

## Contrib

We welcome external sample contributions and TensorFlow contrib API samples to contrib. Please help us improve your experience. See [CONTRIBUTING.md](CONTRIBUTING.md)
