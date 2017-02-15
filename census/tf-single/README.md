# Census: Single node TF vanilla sample (WIP but usable)

This sample uses the [Tensorflow](https://tensorflow.org) low level APIs and
[Google Cloud Machine Learning API](https://cloud.google.com/ml) to demonstrate
the single node TF vanilla version for Census Income Dataset.

## Download the data
Follow the [Census Income
Dataset](https://www.tensorflow.org/tutorials/wide/#reading_the_census_data) link to download the data.

## How to run the code

### Local run
```
python trainer/task.py
usage: task.py [-h] --train_data_path TRAIN_DATA_PATH --eval_data_path
               EVAL_DATA_PATH [--max_steps MAX_STEPS]
```

### gcloud local run
```
gcloud beta ml local train --package-path=trainer --module-name=trainer.task --
--train_data_path TRAIN_DATA_PATH --eval_data_path EVAL_DATA_PATH --max_steps MAX_STEPS
```
