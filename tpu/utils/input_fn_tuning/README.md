# Input Function Tuning

[`input_fn_tuning_job.py`](input_fn_tuning_job.py) is a script for tuning the various parameters used in an `input_fn` built with [TensorFlow Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), such as `cycle_length`



Note: A copy of [`tpu_utils.py`](../tpu_utils.py) is duplicated in this directory for easier import.


## How to use it:

1. Copy (or Git clone) the files in this directory to a VM instance on Compute Engine that has access to TPUs.

1. On the VM instance, run `pip install -r requirements.txt`

1. Modify the definition of your `input_fn` exposing the parameters to be tuned.   

1. Modify `input_fn_tuning_job.py` to include your project ID and TPU location.

1. Run `python input_fn_tuing_job.py`.  
