# TensorFlow Estimator - Trainer Package Template 

The purpose of this repository is to provide a template of how you can package a TensorFlow training model to submit it to Cloud ML Engine. The template makes it easier to organise your code, and to adapt it to your dataset. In more details, the template covers the following functionality:
* Metadata to define your dataset, along with the problem type (classification vs regression).
* Standard implementation of input, parsing, and serving functions.
* Automatic feature columns creation based on the metadata (and normalisation stats).
* Wide & Deep model construction using canned estimators.
* Create custom estimator using standardised model_fn.
* Train, evaluate, and export the model.
* Parameterisation of the experiment.

Although the template provides standard implementation to different functionality, you can customise these parts with your own implementation.


### Repository Structure
1. **template**: The directory includes 1) trainer template with all the python modules to adapt to your data, 2) setup.py, and 3) confg.yaml file for hyper-parameter tuning and specifying the Cloude MLE scale-tier.

2. **scripts**: The directory includes scripts to 1) train the model locally, 2) train the model on Cloud ML Engine, 
and 3) deploy the model on GCP as well as to make prediction (inference) using the deployed model.

3. **examples**: Currently three different examples are included, 1) classification, using the [Census Income](https://archive.ics.uci.edu/ml/datasets/Census+Income) (UCI Machine Learning Repository) dataset, 2) regression, using the [House Pricing](https://www.kaggle.com/apratim87/housingdata/data) (Kaggle) dataset,
and 3) regression, with a custom estimator, using [Natality Baby Weight](https://catalog.data.gov/dataset?tags=birth-weight) (data.gov) dataset.

The examples show how the template is adapted given a dataset. The datasets are found in the examples' folders (under "data" sub-directory).


### Trainer Template Modules

|File Name| Purpose| Do You Need to Change?
|:---|:---|:---
|[metadata.py](https://github.com/ksalama/cloudml-template/blob/master/mle-tf1.4/template/metadata.py)|Defines: 1) task type, 2) input data header, 3) numeric and categorical feature names, 4) target feature name (and labels, for a classification task), and 5) unused feature names. | **Yes**, as you will need to specify the metadata of your dataset. **This might be the only module to change!**
|[input.py](https://github.com/ksalama/cloudml-samples/blob/master/cloudml-template/template/input.py)| Includes: 1) data input functions to read data from csv and tfrecords files, 2) parsing functions to convert csv and tf.example to tensors, 3) function to implement your features custom  processing and creation functionality, and 4) prediction functions (for serving the model) that accepts CSV, JSON, and tf.example instances. | **Maybe**, if you want to implement any custom pre-processing and feature creation during reading data.
|[featurizer.py](https://github.com/ksalama/cloudml-samples/blob/master/cloudml-template/template/featurizer.py)| Creates: 1) tensorflow feature_column(s) based on the dataset metadata (and other extended feature columns, e.g. bucketisation, crossing, embedding, etc.), and 2) deep and wide feature column lists. | **Maybe**, if you want to your feature_column(s) and/or change how deep and wide columns are defined (see next section). 
|[model.py](https://github.com/ksalama/cloudml-samples/blob/master/cloudml-template/template/model.py)|Includes: 1) function to create DNNLinearCombinedRegressor, 2) DNNLinearCombinedClassifier, and 2) function to implement for a custom estimator model_fn.|**No, unless** you want to change something in the estimator, e.g., activation functions, optimizers, etc., or to implement a custom estimator. 
|[task.py](https://github.com/ksalama/cloudml-samples/blob/master/cloudml-template/template/task.py) |Includes: 1 experiment function that executes the model training and evaluation, 2) initialise and parse task arguments (hyper parameters), and 3) Entry point to the trainer. | **No, unless** you want to add/remove parameters, or change parameter default values.

### Suitable for TensorFlow v1.4+
