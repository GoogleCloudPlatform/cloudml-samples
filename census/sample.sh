#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#[TODO] Hardcoded for now, fix when this works
cd estimator

GCS_TRAIN_FILE=gs://cloudml-public/census/data/adult.data.csv
GCS_EVAL_FILE=gs://cloudml-public/census/data/adult.test.csv
CENSUS_DATA=census_data

TRAIN_FILE=$CENSUS_DATA/adult.data.csv
EVAL_FILE=$CENSUS_DATA/adult.test.csv

gsutil cp $GCS_TRAIN_FILE $TRAIN_FILE
gsutil cp $GCS_EVAL_FILE $EVAL_FILE

export TRAIN_STEPS=1000
DATE=`date '+%Y%m%d_%H%M%S'`
export OUTPUT_DIR=census_$DATE

#Local training
python trainer/task.py --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --job-dir $OUTPUT_DIR \
                       --train-steps $TRAIN_STEPS \
                       --eval-steps 100



rm -rf $CENSUS_DATA
rm -rf $OUTPUT_DIR
