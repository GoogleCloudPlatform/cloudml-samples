# Input Function Tuning

[`input_fn_tuning_job.py`](input_fn_tuning_job.py) is a script for tuning the various parameters used in an `input_fn` built with [TensorFlow Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), such as `cycle_length` and `num_parallel_calls`.

Using [`tpu_utils.py`](../tpu_utils.py) the `input_fn_tuning_job.py` script will manage TPU creation and deletion while collecting traces with the [TPU profiler](https://cloud.google.com/tpu/docs/cloud-tpu-tools).

The parameters of each trial are selected by [`scikit-optimize`](https://scikit-optimize.github.io/) to minimize the percentage of time the TPU node spends on waiting for data.


## Sample

Included in this directory is a sample `trainer.py` modeled after the official [ResNet sample](https://github.com/tensorflow/tpu/tree/master/models/official/resnet).


Note: A copy of [`tpu_utils.py`](../tpu_utils.py) is duplicated in this directory for easier import.


## How to use it:

1. Copy (or Git clone) the files in this directory to a VM instance on Compute Engine that has access to TPUs.

1. On the VM instance, run `pip install -r requirements.txt`

1. Modify your training script: (See the included [trainer.py](trainer.py) script for an example)

    * Modify the definition of your `input_fn` exposing the parameters to be tuned.   

    * Modify the main training script to accept command line arguments including the `input_fn` parameters.


1. Run 

    ```
    python input_fn_tuning_job.py \
    --output-dir gs://your-gcs-bucket/path \
    --project-id your-project-id
    ```

1. The accumulated output will be written to `gs://your-gcs-bucket/path/params_scores.yaml`.  The output from a single trial might look like:

    ```
    '1542425224':
    scores:
    - 81.6
    - 75.3
    - 76.9
    score: 77.93333333333332
    input_fn_params:
        transpose_num_parallel_calls: 127
        parallel_interleave_cycle_length: 43
        parallel_interleave_block_length: 1
        tfrecord_dataset_buffer_size: 146350230
        parallel_interleave_prefetch_input_elements: 45
        map_and_batch_num_parallel_calls: 225
        parallel_interleave_buffer_output_elements: 40
        prefetch_buffer_size: 536
        tfrecord_dataset_num_parallel_reads: 114
    ```
