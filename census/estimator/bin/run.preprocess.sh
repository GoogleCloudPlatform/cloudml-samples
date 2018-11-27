#!/bin/bash
# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Convenience script for running preprocessing jobs."""


PROJECT_ID=$(gcloud config list --format 'value(core.project)' 2>/dev/null)
JOB_NAME=preprocessing-${DATE_TIME}-${USER}
DATAFLOW_DIR=gs://census-example/preprocessing/${JOB_NAME}
TRAINING_DATA=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv


python run_preprocessing.py \
  --project_name $PROJECT_ID \
  --job_name $JOB_NAME \
  --job_dir $DATAFLOW_DIR \
  --input_data $TRAINING_DATA \
  --cloud