Flowers: Image-based transfer learning on Cloud ML
--------------------------------------------------

![Build Status](https://storage.googleapis.com/cloudml-samples-test-public/badges/flowers.png)

Build a model to recognize the type of a flower from its image.

This example uses the Flowers dataset to build a customized image classification model via transfer learning and the existent [Inception-v3](https://www.tensorflow.org/tutorials/images/image_recognition) model in order to correctly label different types of flowers using Cloud Machine Learning Engine.

The sample code consist of four parts: 

 - Data preprocessing
 - Model training with the transformed data
 - Model deployment
 - Prediction

- - -

To run this example, first follow instructions for [setting up your environment](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

Also, we use Apache Beam (running on Cloud Dataflow) and PIL to preprocess the images into embeddings, so make sure to install the required packages:
```
pip install -r requirements.txt
```
Then, you may follow the instructions in [sample.sh](https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/flowers/sample.sh).
