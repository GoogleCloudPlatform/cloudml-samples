# Training a scikit-learn model on Cloud ML Engine using the Iris dataset

- - -

### Setup GCP environment
Please follow these [instructions](https://cloud.google.com/ml-engine/docs/scikit/getting-started-training#before_you_begin) to set up your environment with scikit-learn and your Google Cloud Project to train your model on Google Cloud ML Engine.


### Install required packages for training in local environment.
In order to run the sample in your local enviroment, you need to
run the following command to install required python packages.
```
pip install -r requirements.txt
```

### Using `gcloud ml-engine local train`
In the command line, set the following environment variables, replacing [VALUES-IN-BRACKETS] with the appropriate values:
```
TRAINING_PACKAGE_PATH="[YOUR-LOCAL-PATH-TO-TRAINING-PACKAGE]/iris_sklearn_trainer/"
MAIN_TRAINER_MODULE="iris_sklearn_trainer.iris"
OUTPUT_PATH="[YOU-LOCAL-PATH-TO-STORE-OUTPUT-MODEL]"
```
Run your training job locally:
```
gcloud ml-engine local train \
  --job-dir $OUTPUT_PATH \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE
```

### Training on Cloud ML Engine
For your convenience, set the environment variables as below:
```
PROJECT_ID=[YOUR-PROJECT-ID]
BUCKET_ID=[YOUR-BUCKET-ID]
TRAINING_PACKAGE_PATH="[YOUR-LOCAL-PATH-TO-TRAINING-PACKAGE]/iris_sklearn_trainer/"
MAIN_TRAINER_MODULE="iris_sklearn_trainer.iris"
REGION=[REGION]
RUNTIME_VERSION=1.8
PYTHON_VERSION=2.7
SCALE_TIER=BASIC
JOB_NAME="iris_scikit_learn_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME
```
Submit the training job to Cloud ML Engine:
```
gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=$PYTHON_VERSION \
  --scale-tier $SCALE_TIER
```
### Tuning Hyperparameters on Cloud ML Engine
You're also able to do hyperparameter tuning with scikit-learn on Cloud ML Engine. The python package [cloudml-hypertune](https://pypi.org/project/cloudml-hypertune/) is used to help you report hyperparameter tuning objective metrics when running on Cloud ML Engine. cloudml-hypertune can work with any ML framework.

In this sample, `kernel`(kernel type) and `c`(penalty parameter) are the two hyperparameters we hope to tune for the model. Here is the .yaml file for the hyperparameter tuning configuration.
```
# hyperparam.yaml
trainingInput:
  hyperparameters:
    goal: MAXIMIZE
    maxTrials: 20
    maxParallelTrials: 5
    hyperparameterMetricTag: my_metric_tag
    enableTrialEarlyStopping: TRUE 
    params:
    - parameterName: kernel
      type: CATEGORICAL
      categoricalValues: [
          "linear",
          "poly",
          "rbf",
          "sigmoid"
      ]
    - parameterName: c
      type: DOUBLE
      minValue: 0.01
      maxValue: 2
      scaleType: UNIT_LINEAR_SCALE
```
Submit a hyperparameter tuning job on Cloud ML Engine:
```
JOB_NAME="iris_scikit_learn_hptuning_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region us-central1 \
  --runtime-version $RUNTIME_VERSION \
  --python-version $PYTHON_VERSION \
  --scale-tier BASIC \
  --config hyperparam.yaml
```

### Online Predcition
Once you are done training / tuning your model. Check out our guides on how to serve the model for online prediction.
* [Online Prediction with scikit-learn on Google Cloud Machine Learning Engine](https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/sklearn/notebooks/Online%20Prediction%20with%20scikit-learn.ipynb).
* [Getting Started with scikit-learn and XGBoost online predictions](https://cloud.google.com/ml-engine/docs/scikit/quickstart)
* [Online Predictions with scikit-learn Pipelines](https://cloud.google.com/ml-engine/docs/scikit/using-pipelines)
