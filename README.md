# AI Platform Training and Prediction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Welcome to the [AI Platform Training and Prediction](https://cloud.google.com/ml-engine/docs/) sample code repository. This repository contains samples for how to use AI Platform for model training and serving.

## Attention: **Visit our new Vertex AI repo [vertex-ai-samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples)**


## Google Machine Learning Repositories

- [ML on GCP](https://github.com/GoogleCloudPlatform/ml-on-gcp), which has guides on how to bring your code from various ML frameworks to [Google Cloud Platform](https://cloud.google.com/) using things like [Google Compute Engine](https://cloud.google.com/compute/) or [Kubernetes](https://kubernetes.io/).
- [Keras Idiomatic Programmer](https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer) This repository contains content produced by Google Cloud AI Developer Relations for machine learning and artificial intelligence. The content covers a wide spectrum from educational, training, and research, covering from novices, junior/intermediate to advanced.
- [Professional Services](https://github.com/GoogleCloudPlatform/professional-services), common solutions and tools developed by Google Cloud's Professional Services team.

Overview
----------

The repository is organized by tasks: 

 - [Training](#ai-platform-training)
 - [Prediction](#ai-platform-prediction-online-predictions)
 - [Training and Prediction](#complete-guide-model-training-and-prediction-on-ai-platform)
 
Each task can be broken down to general usage `(CPU/GPU)` to specific features: 

 - [Hyperparameter Tuning](#hyperparameter-tuning-hp-tuning)
 - [TPUs](#cloud-tpu) 
 
Scroll down to see what we have available, each task may provide a notebook or code solution. Where the code solution will have a `README` guide and the notebook solution is a full walkthrough. Our code guides are designed to provide you with the code and instructions on how to run the code, but leave you to do the digging, where our notebook tutorials try to walk you through the whole process by having the code available in the notebook throughout the guide.

If you don’t see something for the task you’re trying to complete, please head down to our section [What do you want to see?](#what-do-you-want-to-see)

Setup
-------
For installation instructions and overview, please see [the documentation](https://cloud.google.com/ml-engine/docs/). Please refer to `README.md` in each sample directory for more specific instructions.

Getting Started
---------------
If this is your first time using [AI Platform](https://cloud.google.com/ml-engine/docs/), we suggest you take a look at the [Introduction to AI Platform](https://cloud.google.com/ml-engine/docs/technical-overview) docs to get started.

## AI Platform Training

#### Notebook Tutorial:
* [scikit-learn: Random Forest Classifier](notebooks/scikit-learn/TrainingWithScikitLearnInCMLE.ipynb) - How to train a Random Forest Classifier in scikit-learn using a text based dataset, Census, to predict a person’s income level.
* [XGBoost](notebooks/xgboost/TrainingWithXGBoostInCMLE.ipynb) - How to train an XGBoost model using a text based dataset, Census, to predict a person’s income level.

#### Code Guide:
* [Tensorflow: Linear Classifier with Stochastic Dual Coordinate Ascent (SDCA) Optimizer / Deep Neural Network Classifier](tensorflow/standard/legacy/criteo_tft) - How to train a Linear Classifier with SDCA and a DNN using a text (discrete feature) based dataset, Criteo, to predict how likely a person is to click on an advertisement.
* [Tensorflow: Linear Regression with Stochastic Dual Coordinate Ascent (SDCA) / Deep Neural Network Regressor](tensorflow/standard/legacy/reddit_tft) - How to train a Linear Regressor with SDCA and a DNN using the a text based dataset of Reddit Comments to predict the score of a Reddit thread using a wide and deep model.
* [Tensorflow: ResNet](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) - How to train a model for image recognition using the CIFAR10 dataset to classify image content (training on one CPU, a single host with multiple GPUs, and multiple hosts with CPU or multiple GPUs).

### Cloud TPUs

Tensor Processing Units (TPUs) are Google’s custom-developed ASICs used to accelerate machine-learning workloads. You can run your training jobs on AI Platform, using Cloud TPU.

* [Tensorflow: ResNet](tpu/training/resnet) - Using the ImageNet dataset with Cloud TPUs on AI Platform.
* [Tensorflow: HP Tuning - ResNet](tpu/hptuning/resnet-hptuning) - How to run hyperparameter tuning jobs on AI Platform with Cloud TPUs using TensorFlow's tf.metrics.
* [Tensorflow: Hypertune - ResNet](tpu/hptuning/resnet-hypertune) - How to run hyperparameter tuning jobs on AI Platform with Cloud TPUs using the cloudml-hypertune package.
* [Tensorflow: Cloud TPU Templates](tpu/templates) - A collection of minimal templates that can be run on Cloud TPUs on Compute Engine, Cloud Machine Learning, and Colab.

### Hyperparameter Tuning (HP Tuning)

#### Notebook Tutorial:
* [scikit-learn: Lasso Regressor](notebooks/scikit-learn/HyperparameterTuningWithScikitLearnInCMLE.ipynb) - How to train a Lasso Regressor in scikit-learn using a text based dataset, auto mpg, to predict a car's miles per gallon.
* [XGBoost: XGBRegressor](notebooks/xgboost/HyperparameterTuningWithXGBoostInCMLE.ipynb) - How to train a Regressor in XGBoost using a text based dataset, auto mpg, to predict a car's miles per gallon.

### Containers
* [Keras: Sequential / Dense](tensorflow/containers/unsupported_runtime) - How to train a Keras model using the Nightly Build of TensorFlow on AI Platform using a structured dataset, sonar signals, to predict whether the given sonar signals are bouncing off a metal cylinder or off a cylindrical rock.
* [PyTorch: Deep Neural Network](pytorch/containers/quickstart) - How to train a PyTorch model on AI Platform using a custom container with a image dataset, mnist, to classify handwritten digits.
* [PyTorch: Sequential](pytorch/containers/custom_container) - How to train a PyTorch model on AI Platform using a custom container with a structured dataset, sonar signals, to predict whether the given sonar signals are bouncing off a metal cylinder or off a cylindrical rock.
* [PyTorch: Sequential / HP Tuning](pytorch/containers/hp_tuning) - How to train a PyTorch model on AI Platform using a custom container and Hyperparameter Tuning with a structured dataset, sonar signals, to predict whether the given sonar signals are bouncing off a metal cylinder or off a cylindrical rock.

## AI Platform Prediction (Online Predictions)

#### Notebook Tutorial:
* [scikit-learn: Model Serving](notebooks/scikit-learn/OnlinePredictionWithScikitLearnInCMLE.ipynb) - How to train a Random Forest Classifier in scikit-learn on your local machine using a text based dataset, Census, to predict a person’s income level and deploy it on AI Platform to create predictions.
* [XGBoost: Model Serving](notebooks/xgboost/OnlinePredictionWithXGBoostInCMLE.ipynb) -  How to train an XGBoost model on your local machine using a text based dataset, Census, to predict a person’s income level and deploy it on AI Platform to create predictions.

## Complete Guide: Model Training and Prediction on AI Platform

#### Code Guide:
* [Tensorflow: Deep Neural Network Regressor](molecules) - How to train a DNN on a text based molecular dataset from Kaggle to predict the molecular energy.
* [Tensorflow: Softmax / Fully-connected layer](flowers) - How to train a fully connected model with softmax using an image dataset of flowers to recognize the type of a flower from its image.

### Hyperparameter Tuning (HP Tuning)

#### Code Guide:

* [Keras: Sequential / Dense](census/keras) - [Keras](https://keras.io/) - How to train a Keras sequential and Dense model using a text based dataset, Census, to predict a person’s income level using a single node model.
* [Tensorflow Pre-made Estimator: Deep Neural Network Linear Combined Classifier](census/estimator) -How to train a DNN using Tensorflow’s Pre-made Estimators using a text based dataset, Census, to predict a person’s income level. [TensorFlow Pre-made Estimator](https://www.tensorflow.org/programmers_guide/estimators#pre-made_estimators), an estimator is “a high-level TensorFlow API that greatly simplifies machine learning programming.”
* [Tensorflow Custom Estimator: Deep Neural Network](census/customestimator) - How to train a DNN using Tensorflow’s Custom Estimators using a text based dataset, Census, to predict a person’s income level. [TensorFlow Custom Estimator](https://www.tensorflow.org/programmers_guide/estimators#custom_estimators), which is when you write your own model function. 
* [Tensorflow: Deep Neural Network](census/tensorflowcore) - How to train a DNN using TensorFlow’s low level APIs to create your DNN model on a single node using a text based dataset, Census, to predict a person’s income level.
* [Tensorflow: Matrix Factorization / Deep Neural Network with Softmax](tensorflow/standard/legacy/movielens) - How to train a Matrix Factorization and DNN with Softmax using a text based dataset, MovieLens 20M, to make movie recommendations.

Templates
---------

* [TensorFlow Estimator Trainer Package Template](cloudml-template) - When training a Tensorflow model, you have to create a trainer package, here we have a template that simplifies creating a trainer package for AI Platform. Take a look at this list with some introductory [examples](cloudml-template/examples/). 

* [Tensorflow: Cloud TPU Templates](tpu/templates) - A collection of minimal templates that can be run on Cloud TPUs on Compute Engine, Cloud Machine Learning, and Colab.

* [Scikit-learn Pipelines Trainer Package Template](sklearn/sklearn-template/template) - You can use this as starter code to develop a scikit-learn model for training and prediction on AI Platform. [Examples](sklearn/sklearn-template/examples) to be added.


Additional Resources
--------------------

- ### Cloud TPU 
Please see the [Cloud TPU guide](tpu) for how to use Cloud TPU.

- ### Google Samples

  * [Genomics Ancestry Inference](https://github.com/googlegenomics/cloudml-examples) - Genomics ancestry inference using 1000 Genomes dataset


## What do you want to see?

If you came looking for a sample we don’t have, please file an issue using the [Sample / Feature Request](https://github.com/GoogleCloudPlatform/cloudml-samples/issues/new?template=sample-feature-request.md) template on this repository. Please provide as much detail as possible about the AI Platform sample you were looking for, what framework (Tensorflow, Keras, scikit-learn, XGBoost, PyTorch...), the type of model, and what kind of dataset you were hoping to use! 

Jump below if you want to contribute and add that missing sample.

How to contribute?
------------------

We welcome external sample contributions! To learn more about contributing new samples, checkout our [CONTRIBUTING.md](CONTRIBUTING.md) guide. Please feel free to add new samples that are built in notebook form or code form with a README guide. 

Want to contribute but don't have an idea? Check out our [Sample Request Page](https://github.com/GoogleCloudPlatform/cloudml-samples/issues?q=is%3Aissue+is%3Aopen+label%3ASAMPLE_REQUEST) and assign the issue to yourself so we know you're working on it!

Documentation
-------------

We host AI Platform documentation [here](https://cloud.google.com/ml-engine/docs/)


Disclaimer
-------------

The content in the `CloudML-Samples` repository is not officially maintained by Google.
