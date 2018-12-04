# Overview
This code implements image-based transfer learning on Cloud ML
In this tutorial you will walk through and results you will monitor consist of four parts: data preprocessing, model training with the transformed data,
model deployment, and prediction request steps. All parts will be completed in the cloud.

#
* **Data description**

This tutorial uses the Flowers [dataset](https://storage.cloud.google.com/cloud-ml-data/img/flower_photos/all_data.csv) to build a customized image classification model via transfer learning and the existent [Inception-v3 model](https://www.tensorflow.org/tutorials/image_recognition) 
in order to correctly label different types of flowers using Cloud Machine Learning Engine.

* **Disclaimer**

This dataset is provided by a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.

* **Setup and test your GCP environment**

The best way to setup your GCP project is to use this section in this
[tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#set-up-your-gcp-project).

* **Environment setup:**

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
locally without changing global python packages on your system.

There are two options for the virtual environments:

*   Install [Virtualenv](https://virtualenv.pypa.io/en/stable/) 
    *   Create virtual environment `virtualenv myvirtualenv`
    *   Activate env `source myvirtualenv/bin/activate`
*   Install [Miniconda](https://conda.io/miniconda.html)
    *   Create conda environment `conda create --name myvirtualenv python=2.7`
    *   Activate env `source activate myvirtualenv`

* **Install dependencies**

Install the python dependencies. `pip install --upgrade -r requirements.txt`

#

* **How to satisfy Cloud ML Engine project structure requirements**

Follow [this](https://cloud.google.com/ml-engine/docs/tensorflow/packaging-trainer#project-structure) guide to structure your training application.



# Data processing

You will run sample code in order to preprocess data with Cloud Dataflow and then use that transformed data to train a model with Cloud ML Engine. You will then deploy the trained model to Cloud ML Engine and test the model by sending a prediction request to it.

In this sample dataset you only have a small set of images (~3,600). Without more data it isn’t possible to use machine learning techniques to adequately train an accurate classification model from scratch. Instead, you’ll use an approach called transfer learning. In transfer learning you use a pre-trained model to extract image features that you will use to train a new classifier. In this tutorial in particular you’ll use a pre-trained model called Inception.

```
export PROJECT=$(gcloud config list project --format "value(core.project)")
export JOB_ID="flowers_${USER}_$(date +%Y%m%d_%H%M%S)"
export BUCKET="gs://${PROJECT}-ml"
export GCS_PATH="${BUCKET}/${USER}/${JOB_ID}"
export DICT_FILE=gs://cloud-samples-data/ml-engine/flowers/dict.txt

export MODEL_NAME=flowers
export VERSION_NAME=v1
```

* **Use DataFlow to preprocess dataset**

Takes about 30 mins to preprocess everything.  We serialize the two
preprocess.py synchronous calls just for shell scripting ease; you could use
`--runner DataflowRunner` to run them asynchronously.  Typically,
the total worker time is higher when running on Cloud instead of your local
machine due to increased network traffic and the use of more cost efficient
CPU's.  Check progress [here](https://console.cloud.google.com/dataflow)

Pre-process training

```
python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "gs://cloud-samples-data/ml-engine/flowers/train_set.csv" \
  --output_path "${GCS_PATH}/preproc/train" \
  --cloud
```  
  
Pre-process evaluation

```
python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "gs://cloud-samples-data/ml-engine/flowers/eval_set.csv" \
  --output_path "${GCS_PATH}/preproc/eval" \
  --cloud
```

  

# Training

* **Google Cloud ML Engine**


* **Run in Google Cloud ML Engine**

Training on CloudML is quick after preprocessing.  If you ran the above
commands asynchronously, make sure they have completed before calling this one.


* **Run in Google Cloud ML Engine:**

```
gcloud ml-engine jobs submit training "$JOB_ID" \
  --stream-logs \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --runtime-version=1.10 \
  -- \
  --output_path "${GCS_PATH}/training" \
  --eval_data_paths "${GCS_PATH}/preproc/eval*" \
  --train_data_paths "${GCS_PATH}/preproc/train*"
```

* **Monitor with TensorBoard:**

```
tensorboard --logdir=${GCS_PATH}/training
```


# Prediction

Remove the model and its version, make sure no error is reported if model does not exist.
```
gcloud ml-engine versions delete $VERSION_NAME --model=$MODEL_NAME -q --verbosity none
gcloud ml-engine models delete $MODEL_NAME -q --verbosity none
```

Once your training job has finished, you can use the exported model to create a prediction server. To do this you first create a model:

```
gcloud ml-engine models create "$MODEL_NAME" \
  --regions us-central1
```


Each unique Tensorflow graph--with all the information it needs to execute--
corresponds to a "version".  Creating a version actually deploys our
Tensorflow graph to a Cloud instance, and gets is ready to serve (predict).

```
gcloud ml-engine versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "${GCS_PATH}/training/model" \
  --runtime-version=1.10
```

Models do not need a default version, but its a great way move your production
service from one version to another with a single gcloud command.

```
gcloud ml-engine versions set-default "$VERSION_NAME" --model "$MODEL_NAME"
```

* **Run Online Predictions**

You can now send prediction requests to the API. To test this out you can use the `gcloud ml-engine predict` tool:

Download a daisy so we can test online predictions.
```
gsutil cp \
  gs://cloud-samples-data/ml-engine/flowers/daisy/100080576_f52e8ee070_n.jpg \
  daisy.jpg
```

Since the image is passed via JSON, we have to encode the JPEG string first.

```
python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"key":"0", "image_bytes": {"b64": img}})' daisy.jpg &> request.json
```

Test online prediction

```
gcloud ml-engine predict --model ${MODEL_NAME} --json-instances request.json
```

You should see a response with the predicted labels of the examples!


## References

[Flowers tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/flowers-tutorial)
