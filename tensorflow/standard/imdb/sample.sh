#!/bin/bash
# Copyright 2018 Google Inc. All Rights Reserved.
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


run_script_local() {

	cd $1
	echo "Running '$1' code sample."
	GCS_TRAIN_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb.npz
	GCS_WORD_INDEX_FILE=gs://cloud-samples-data/ml-engine/imdb/imdb_word_index.json
    DATA=imdb_data
	TRAIN_FILE=$DATA/imdb.npz
	WORD_INDEX_FILE=$DATA/imdb_word_index.json

	gsutil cp $GCS_TRAIN_FILE $TRAIN_FILE
	gsutil cp $GCS_WORD_INDEX_FILE $WORD_INDEX_FILE

	DATE=`date '+%Y%m%d_%H%M%S'`
	JOB_DIR=iris_$DATE

	# Local training.
	python -m trainer.task --train-file=$TRAIN_FILE \
                       --word-index-file=$WORD_INDEX_FILE \
                       --job-dir=$JOB_DIR

	if [ $? = 0 ]; then
		echo "Python script succeeded"
		rm -rf $DATA
		rm -rf $JOB_DIR
		cd ..
		return 0
	fi
	echo "Python script failed"
	return 1
}

run_script_local tensorflow/keras
if [ $? = 1 ]; then
	exit 1
fi