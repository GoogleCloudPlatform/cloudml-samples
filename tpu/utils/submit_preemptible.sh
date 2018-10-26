# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


export TPU_NAME=$1
export SCRIPT_NAME=your_train_task.py
export MODEL_DIR=gs://your-gcs-bucket

python $SCRIPT_NAME \
        --tpu=$TPU_NAME \
        --model_dir=$MODEL_DIR \
        --mode=train \
        --skip_host_call=False \
        --train_batch_size=6144 \
        --train_steps=6144 \
        --num_cores=128 \
        --iterations_per_loop=256
