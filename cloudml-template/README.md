# Cloud ML Engine - TF Trainer Package Template

The purpose of this repository is to provide a template of how you can package a TensorFlow training model to submit it to Cloud ML Engine. 
The template makes it easier to organise your code, and to adapt it to your dataset. In more details, the template covers the following functionality:
* Metadata to define your dataset, along with the problem type (classification vs regression)
* Standard implementation of input, parsing, and serving functions
* Feature columns creation based on the metadata (and normalisation stats)
* Wide and Deep model creation using canned estimators
* Create custom estimator using standardised model_fn
* Train, evaluate, and export the model
* Parameterisation of the experiment

Although the template provides standard implementation to different functionality, you can customise these parts with your own implementation.

## Latest TF Version: Tensorflow v1.4

### Repository Structure
1. **template**: includes all the python module files to adapt to your data to build the ML trainer.

2. **examples**: currently 3 examples are included, 1) classification, using the [Census Income](https://archive.ics.uci.edu/ml/datasets/Census+Income) dataset, 2) regression, using the [House Pricing](https://www.kaggle.com/apratim87/housingdata/data) dataset,
and 3) regression, with a custom estimator, using [Natality Baby Weight](https://catalog.data.gov/dataset?tags=birth-weight) dataset.
The examples show how the template is adapted given a dataset. The datasets are found in the examples' folders.


3. **scripts**: includes scripts to 1) train the model locally, 2) train the model on Cloud ML Engine, 
and 3) deploy the model on GCP as well as to make prediction (inference) using the deployed model.

4. **mle_inference.py**: a python code to perform prediction (inference ) via calling  the deployed model API

### Trainer Template Modules


|File Name| Purpose| Do You Need to Change?
|:---|:---|:---
|[metadata.py](https://github.com/ksalama/cloudml-template/blob/master/mle-tf1.4/template/metadata.py)|Defines: 1) task type, 2) input data header, 3) numeric and categorical feature names, 4) target feature name (and labels, for a classification task), and 5) unused feature names. | **Yes**, as you will need to specify the metadata of your dataset. **This might be the only module to change!**
|[input.py](https://github.com/ksalama/cloudml-template/blob/master/mle-tf1.4/template/input.py)| Includes: 1) data input functions to read data from csv and tfrecords files, 2) parsing functions to convert csv and tf.example to tensors, 3) function to implement your features custom  processing and creation functionality, and 4) prediction functions (for serving the model) that accepts CSV, JSON, and tf.example instances. | **Maybe**, if you want to implement any custom pre-processing and feature creation during reading data.
|[featurizer.py](https://github.com/ksalama/cloudml-template/blob/master/mle-tf1.4/template/featurizer.py)| Creates: 1) tensorflow feature_column(s) based on the dataset metadata (and other extended feature columns, e.g. bucketisation, crossing, embedding, etc.), and 2) deep and wide feature column lists. | **Maybe**, if you want to your feature_column(s) and/or change how deep and wide columns are defined (see next section). 
|[model.py](https://github.com/ksalama/cloudml-template/blob/master/mle-tf1.4/template/model.py)|Includes: 1) function to create DNNLinearCombinedRegressor, 2) DNNLinearCombinedClassifier, and 2) function to implement for a custom estimator model_fn.|**No, unless** you want to change something in the estimator, e.g., activation functions, optimizers, etc., or to implement a custom estimator. 
|[task.py](https://github.com/ksalama/cloudml-template/blob/master/mle-tf1.4/template/task.py) |Includes: 1 experiment function that executes the model training and evaluation, 2) initialise and parse task arguments (hyper parameters), and 3) Entry point to the trainer. | **No, unless** you want to add/remove parameters, or change parameter default values.


### Featurizer - defining deep and wide columns

* numeric_columns + embedding_columns &rarr; **dense_columns** (int and float features)
* categorical_columns_with_identity + categorical_columns_with_vocabolary_list + bucketized_columns &rarr; **categorical_columns** (low-cardinality categorical features)
* categorical_columns_with_hash_buckets + crossed_columns &rarr; **sparse_columns** (high-cardinality categorical features)

* categorical_columns &rarr; **indicator_columns** (one-hot encoding)

* **deep_columns** = *dense_columns* + *indicator_columns*
* **wide_columns** = *categorical_columns* + *sparse_columns*
