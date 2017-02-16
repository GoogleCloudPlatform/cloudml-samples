# Census: Single node TF vanilla sample (WIP but usable)

This sample uses the [TensorFlow](https://tensorflow.org) low level APIs and
[Google Cloud Machine Learning API](https://cloud.google.com/ml) to demonstrate
the single node TF vanilla version for Census Income Dataset.

## Download the data
Follow the [Census Income
Dataset](https://www.tensorflow.org/tutorials/wide/#reading_the_census_data) link to download the data.


## Install dependencies
Install the following dependencies:

 * Install [Virtual](https://virtualenv.pypa.io/en/stable/) env
   * Create virtual environment
      ```
      virtualenv env
      source env/bin/activate
      ```
 * Install [TensorFlow](https://www.tensorflow.org/install/)
 * Install [Pandas](http://pandas.pydata.org/pandas-docs/stable/install.html#installing-from-pypi)
  

## How to run the code

### Local run
```
python trainer/task.py --train_data_path TRAIN_DATA_PATH
                       --eval_data_path EVAL_DATA_PATH
                       [--max_steps MAX_STEPS]
```

### gcloud local run
```
gcloud beta ml local train --package-path=trainer
                           --module-name=trainer.task
                           -- --train_data_path TRAIN_DATA_PATH
                           --eval_data_path EVAL_DATA_PATH
                           [--max_steps MAX_STEPS]
```
