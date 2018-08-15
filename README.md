# Google Cloud Machine Learning Engine

Welcome to the Google Cloud Machine Learning Engine (Cloud ML Engine) repository. This repository contains samples for how to use Cloud ML Engine for model training and serving.

Note: If you’re looking for our guides on how to do Machine Learning on Google Cloud Platform (GCP) without using Cloud ML Engine, please checkout our other repository: [ML on GCP](https://github.com/GoogleCloudPlatform/ml-on-gcp), which has guides on how to bring your code from various ML frameworks to [Google Cloud Platform](https://cloud.google.com/) using things like Google Compute Engine or Kubernetes .

# Setup

For installation instructions and overview, please see [the documentation](https://cloud.google.com/ml-engine/docs/). Please refer to **README.md** in each sample directory for more specific instructions.

# Getting Started

TODO: Point back to docs

# Overview
The repository is organized by tasks, scroll down to see what we have available, each task may provide a code or notebook solution. Where the code solution will have a README guide and the notebook solution is a full walkthrough. Our traditional code solution guides are designed to provide you with the code and instructions on how to run the code, but leave you to do the digging, where our notebook tutorials try to walk you through the whole process.

If you don’t see something for the task you’re trying to complete, please head down to our section “What do you want to see?”

# Model Training with Cloud ML Engine

Notebook Tutorial:
* [scikit-learn: Random Forest Classifier](sklearn/notebooks/ml_engine_training.ipynb) - How to train a Random Forest Classifier in scikit-learn using a text based dataset, Census, to predict a person’s income level.
 * [XGBoost](xgboost/notebooks/XGBoost%20training%20with%20ML%20Engine.ipynb) - How to train an XGBoost model using a text based dataset, Census, to predict a person’s income level.

Traditional Code Guide:
* [Tensorflow: Linear Classifier with Stochastic Dual Coordinate Ascent (SDCA) optimizer / Deep Neural Network Classifier](criteo_tft) - How to train a Linear Classifier with SDCA and a DNN using a text based dataset, Criteo, to predict how likely a person is to click on an advertisement.
* [Tensorflow: Linear Regression with Stochastic Dual Coordinate Ascent (SDCA) / Deep Neural Network Regressor](reddit_tft) - How to train a Linear Regressor with SDCA and a DNN using the a text based dataset of Reddit Comments to predict the score of a Reddit thread using a wide and deep model.
* [Tensorflow: ResNet](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) - How to train a model for image recognition using the Cifar10 dataset to classify image content (training on one CPU, a single host with multiple GPUs, and multiple hosts with CPU or multiple GPUs).

### Cloud TPUs on ML Engine

Tensor Processing Units (TPUs) are Google’s custom-developed ASICs used to accelerate machine-learning workloads. You can run your training jobs on Cloud Machine Learning Engine, using Cloud TPU.

* [Tensorflow: ResNet](tpu/training/resnet) - Using the ResNet-50 dataset with Cloud TPUs on ML Engine.
* [Tensorflow: HP Tuning - ResNet](tpu/hptuning/resent-hptuning) - How to run hyperparameter tuning jobs on Cloud Machine Learning Engine with Cloud TPUs using TensorFlow's tf.metrics.
* [Tensorflow: Hypertune - ResNet](tpu/hptuning/resent-hypertune) - How to run hyperparameter tuning jobs on Cloud Machine Learning Engine with Cloud TPUs using the cloudml-hypertune package.

# Model Serving (Online Predictions)

Notebook Tutorial:
* [scikit-learn: Model Serving](sklearn/notebooks/Online%20Prediction%20with%20scikit-learn.ipynb) - How to train a Random Forest Classifier in scikit-learn on your local machine using a text based dataset, Census, to predict a person’s income level and deploy it on Cloud ML Engine to create predictions.
* [XGBoost: Model Serving](xgboost/notebooks/Online%20Prediction%20with%20XGBoost.ipynb) -  How to train an XGBoost model on your local machine using a text based dataset, Census, to predict a person’s income level and deploy it on Cloud ML Engine to create predictions.

Traditional Code Guide:

# Complete Guide: Model Training and Serving on ML Engine

Traditional Code Guide:
* [Tensorflow: Deep Neural Network Regressor](molecules) - How to train a DNN on a text based molecular dataset from Kaggle to predict predict the molecular energy.
* [Tensorflow: Softmax / Fully-connected layer](flowers) - How to train a fully connected model with softmax using an image dataset of flowers to recognize the type of a flower from its image.

### Hyperparameter Tuning (HP Tuning)

Traditional Code Guide:
* [Keras: Sequential / Dense](census/keras) - [Keras](https://keras.io/) - How to train a Keras sequential and Dense model using a text based dataset, Census, to predict a person’s income level using a single node model.
* [Tensorflow Pre-made Estimator: Deep Neural Network Linear Combined Classifier](census/estimator) -How to train a DNN using Tensorflow’s Pre-made Estimators using a text based dataset, Census, to predict a person’s income level. [TensorFlow Pre-made Estimator](https://www.tensorflow.org/programmers_guide/estimators#pre-made_estimators), an estimator is “a high-level TensorFlow API that greatly simplifies machine learning programming.”
* [Tensorflow Custom Estimator: Deep Neural Network](census/customestimator) - How to train a DNN using Tensorflow’s Custom Estimators using a text based dataset, Census, to predict a person’s income level. [TensorFlow Custom Estimator](https://www.tensorflow.org/programmers_guide/estimators#custom_estimators), which is when you write your own model function. 
* [Tensorflow: Deep Neural Network](census/tensorflowcore) - How to train a DNN using TensorFlow’s low level APIs to create your DNN model on a single node using a text based dataset, Census, to predict a person’s income level.
* [Tensorflow: Matrix Factorization / Deep Neural Network with Softmax](movielens) - How to train a Matrix Factorization and DNN with Softmax using a text based dataset, MovieLens 20M, to make movie recommendations.

# Cloud TPU

Please see the [guide here](CLOUD_TPU_README.md) for how to use Cloud TPU. 


# Resources

* [TensorFlow Estimator Trainer Package Template](cloudml-template) - When training a Tensorflow model, you have to create a trainer package, here we have a template that simplifies creating a trainer package for Cloud ML Engine.
* Tensorflow to TF-Lite

## Google Samples

* [Genomics Ancestry Inference](https://github.com/googlegenomics/cloudml-examples) - Genomics ancestry inference using 1000 Genomes dataset

# What do you want to see?

If you came looking for a sample we don’t have, please file an issue on this repository making a request for what you were looking for and provide as much detail as possible! Jump below if you want to contribute and add that missing sample.

# How to Contribute

We welcome external sample contributions! To learn more about contributing new samples, checkout our [CONTRIBUTING.md](CONTRIBUTING.md) guide. Please feel free to add new samples that are built in notebook form or standard code form with a README guide. 
