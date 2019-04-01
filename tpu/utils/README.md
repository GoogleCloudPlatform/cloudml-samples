# Utilities

This directory contains helpful utilities to use [Cloud TPU](https://cloud.google.com/tpu/).


## TPU Utilities

[`tpu_utils.py`](tpu_utils.py) contains Python wrapper functions for the [Cloud TPU REST API](https://cloud.google.com/tpu/docs/reference/rest/) to manage the TPU resources.  This allows programatically creating, deleting, and monitoring TPU nodes.


## TPU Survival Training

[TPU survival training](survival/) is a tool for resuming training when a [preemptible TPU](https://cloud.google.com/tpu/docs/preemptible) is preempted during a training session.


## Input Function Tuning

[Input function tuning](input_fn_tuning/) is a tool for automatically tuning the various parameters of the `input_fn` using [Scikit-Optimize](https://scikit-optimize.github.io/).
