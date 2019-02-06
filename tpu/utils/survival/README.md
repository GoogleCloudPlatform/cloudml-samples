# TPU Survival Training

[`tpu_survival_job.py`](tpu_survival_job.py) is a simple script for resuming training when a [preemptible TPU](https://cloud.google.com/tpu/docs/preemptible) is preempted during a training session.

Note: A copy of [`tpu_utils.py`](../tpu_utils.py) is duplicated in this directory for easier import.


## How to use it:

1. Copy (or Git clone) the files in this directory to a VM instance on Compute Engine that has access to TPUs.

1. Modify the `submit_preemptible.sh` script for your training job.  

1. Modify `tpu_survival_job.py` to include your project ID and TPU location.

1. Run `python tpu_survival_job.py`.  Each time the TPU is preempted a new one will be created and the training job resumed from the last saved checkpoint.
