# Utilities

This directory contains helpful utilities to use [Cloud TPU](https://cloud.google.com/tpu/).


## TPU Utilities

[`tpu_utils.py`](tpu_utils.py) contains Python wrapper functions for the [Cloud TPU REST API](https://cloud.google.com/tpu/docs/reference/rest/) to manage the TPU resources.  This allows programatically creating, deleting, and monitoring TPU nodes.


## TPU Survival Training

[`tpu_survival_job.py`](survival/tpu_survival_job.py) is a script for resuming training when a [preemptible TPU](https://cloud.google.com/tpu/docs/preemptible) is preempted during a training session.


## Input Function Tuning

[`input_fn_tuning_job.py`](input_fn_tuning/input_fn_tuning_job.py) is a tool for tuning the various parameters of the `input_fn` using Bayesian Optimization.
