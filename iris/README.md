Iris: End-to-end Cloud Machine Learning API sample (DEPRECATED)
--------------------------------------------------

![status: inactive](https://img.shields.io/badge/status-inactive-red.svg)
This sample has been deprecated. It has not been tested against the newest version of TensorFlow. When running `gcloud ml-engine` commands be sure to use `--runtime-version 0.12`. For a maintained sample check out [The Census Example](../census)

This sample uses the [Google Cloud Machine Learning API](https://cloud.google.com/ml), [Tensorflow](https://tensorflow.org), [Apache Beam](https://cloud.google.com/dataflow), and the provided Cloud ML SDK to,

* Preprocess data
* Start a Cloud ML API Training Job
* Create a Cloud ML model, from the output
* Start a Cloud ML Prediction service
* Predict values of new instances.

To run this example, first follow instructions for [setting up your environment](https://cloud.google.com/ml/docs/how-tos/getting-set-up), and preprocess the data using preprocess.py, then you may follow instructions in the [MNIST quickstarts](https://cloud.google.com/ml/docs/quickstarts/training) replacing relevant filepaths with ones which refer to this directory.

Alternatively you may run the example end-to-end in Dataflow. Run,

```
python pipeline.py --help
```

for the full list of required and optional command line arguments.
