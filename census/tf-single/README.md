# Census: Single node TF vanilla sample

This sample uses the [TensorFlow](https://tensorflow.org) low level APIs and
[Google Cloud Machine Learning API](https://cloud.google.com/ml) to demonstrate
the single node TF vanilla version for Census Income Dataset.

## Download the data
Follow the [Census Income
Dataset](https://www.tensorflow.org/tutorials/wide/#reading_the_census_data) link to download the data. You can also download directly from [here](https://archive.ics.uci.edu/ml/datasets/Census+Income).

 * Training file is `adult.data`
 * Evaluation file is `adult.test` 


## Virtual environment
There are two options for the virtual environments:
 * Install [Virtual](https://virtualenv.pypa.io/en/stable/) env
   * Create virtual environment `virtualenv single-tf`
   * Activate env `source single-tf/bin/activate`
 * Install [Miniconda](https://conda.io/miniconda.html)
   * Create conda environment `conda create --name single-tf python=2.7`
   * Activate env `source activate single-tf`


## Install dependencies
Install the following dependencies:
 * Install [TensorFlow](https://www.tensorflow.org/install/)
 * Install [Pandas](http://pandas.pydata.org/pandas-docs/stable/install.html#installing-from-pypi)
  

## How to run the code
### Help
```
python trainer/task.py -h
```

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

## Accuracy
You should see an accuracy of `83.25%` for default number of training steps.
