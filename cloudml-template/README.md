# TensorFlow Estimator - Trainer Package Template

The purpose of this repository is to provide a template of how you can package a TensorFlow training model to submit it to AI Platform. The template makes it easier to organise your code, and to adapt it to your dataset. In more details, the template covers the following functionality:
* Metadata to define your dataset, along with the problem type (classification vs regression).
* Standard implementation of input, parsing, and serving functions.
* Automatic feature columns creation based on the metadata (and normalisation stats).
* Wide & Deep model construction using canned estimators.
* Train, evaluate, and export the model.
* Parameterisation of the experiment.

Although the template provides standard implementation to different functionality, you can customise these parts with your own implementation.


### Repository Structure

1. **[template](template)**: The directory includes: 
    1) trainer template with all the python modules to adapt to your data.
    2) ```setup.py```.
    3) ```config.yaml``` file for hyper-parameter tuning and specifying the AI Platform scale-tier.
    4) ```inference.py``` python (sample) script to perform prediction using a deployed model's API on AI Platform.

2. **[scripts](scripts)**: The directory includes command-line scripts to:
    1) Train the model locally.
    2) Train the model on AI Platform. 
    3) Deploy the model on GCP as well as to make prediction (inference) using the deployed model.

3. **[examples](examples)**: Four different examples are included:
    1. [Census Income](https://archive.ics.uci.edu/ml/datasets/Census+Income) (UCI Machine Learning Repository) dataset (**classification**).
    2. [Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29) (UCI Machine Learning Repository) dataset (**classification**).
    3. [House Pricing](https://www.kaggle.com/apratim87/housingdata/data) (Kaggle) dataset (**regression**).
    4. [Natality Baby Weight](https://catalog.data.gov/dataset?tags=birth-weight) (data.gov) dataset (**regression**).


The examples show how the template is adapted given a dataset. The datasets are found in the examples' folders (under "data" sub-directory).


### Trainer Template Modules

|File Name| Purpose| Do You Need to Change?
|:---|:---|:---
|[metadata.py](template/trainer/metadata.py)|Defines: 1) task type, 2) input data header, 3) numeric and categorical feature names, and 4) target feature name (and labels, for a classification task) | **Yes**, as you will need to specify the metadata of your dataset. **This might be the only module to change!**
|[inputs.py](template/trainer/inputs.py)| Includes: 1) data input functions to read data from csv and tfrecords files, 2) parsing functions to convert csv and tf.example to tensors, 3) function to implement your custom  features processing and creation functionality, and 4) prediction functions (for serving the model) that accepts CSV, JSON, and tf.example instances. | **Maybe**, if you want to implement any custom pre-processing and feature creation during reading data.
|[featurizer.py](template/trainer/featurizer.py)| Creates: 1) TensorFlow feature_column(s) based on the dataset metadata (and other extended feature columns, e.g. bucketisation, crossing, embedding, etc.), and 2) deep and wide feature column lists. | **Maybe**, if you want to change your feature_column(s) and/or change how deep and wide columns are defined. 
|[model.py](template/trainer/model.py)|Includes: 1) function to create DNNLinearCombinedRegressor, and 2) DNNLinearCombinedClassifier.|**No, unless** you want to change something in the estimator, e.g., activation functions, optimizers, etc.. 
|[experiment.py](template/trainer/task.py) |Runs the model training and evaluation experiment, and exports the final model. | **No, unless** you want to add/remove parameters, or change parameter default values.
|[task.py](template/trainer/task.py) |Includes: 1) Initialise and parse task arguments (hyper parameters), and 2) Entry point to the trainer. | **No, unless** you want to add/remove parameters, or change parameter default values.


### Suitable for TensorFlow v1.13.1+
