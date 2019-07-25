# Data science with R on GCP

R is one of the most widely used programming languages for statistical modeling, which has a large and active community of data scientists and Machine Learning (ML) professional. With over 10,000 packages in the open-source repository of CRAN, R caters to all statistical data analysis applications, ML, and visualisation. According to IEEE Spectrum ranking, it is one of the top 10 programming languages in 2018, with steady growth in the last two decades, due to its expressiveness of its syntax, and comprehensibility of its data and ML libraries.


This multi-part R on GCP tutorial series covers the following topics, each in a separate notebook
1. [Exploratory data analysis](01_EDA-with-R-and-BigQuery.ipynb). - using R and [BigQuery](https://cloud.google.com/bigquery/docs)
2. [Training and serving Tensorflow models](02-Training-Serving-TF-Models.ipynb) - using [AI Platform](https://cloud.google.com/ml-engine/docs/tensorflow/)
3. [Training and serving CARET models](03-Training-Serving-CARET-Models.ipynb) - using AI Platform [Training with Custom Containers](https://cloud.google.com/ml-engine/docs/custom-containers-training), and [Cloud Run](https://cloud.google.com/run/docs/) for serving.

In addition to the notebooks, the supporting R source code and Dockerfiles are includes in the [src](src) directory, which includes:
1. [src/tensorflow](src/tensorflow): for training the TensorFlow model on AI Platform Training.
2. [src/caret](src/caret): for training a CARET model on AI Platform with custom container, and serving the CARET model as a Web API on Cloud Run. 

The notebooks use the R 3.5.3 kernel. You can run these notebooks in [AI Platform Notebooks](https://cloud.google.com/ai-platform-notebooks), using the following steps:

1. [Create an AI Platform Notebook Instance](https://console.google.com/mlengine/notebooks/create-instance)
2. Fill the form shown below, by providing you Insance name, region, Zone, Machine Type, and Boot disk options
3. Make sure that you select **R 3.5.3** as the Framework
4. Click **CREATE**
5. After creating the AI Platform Notebook Instance, click on **OPEN JUPYTERLAB link**, to open the JupyterLab environment in your browser.
6. Launch a terminal tab
7. Clone the cloudml-samples Github repository by executing the following shell command:

``` bash
git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git
```

8. When the command finishes, you will notice that a cloudml-sample folder appeared in the left pan. Navigate to cloudml-samples>notebooks>R


9. Run the following commands to create and download the service account key:

``` bash
cd cloudml-samples/notebooks/R 
gcloud config set project [YOUR-PROJECT-ID]
gcloud auth list 
```
10. Copy the service account listed, and run the following command by replacing SERVICE-ACCOUNT with the service account you copied from the list: 

``` bash
cloud iam service-accounts keys create service-account-key.json --iam-account=[SERVICE-ACCOUNT]
```

