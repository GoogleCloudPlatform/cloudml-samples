# Template for training a Scikit-learn Model on Cloud ML Engine

```
Template              
    |__ config.yaml  (file for CloudML configurations)
    |__ setup.py 
    |__ trainer  (trainer package)
        |__ __init__.py
        |__ launch_demo.py  (end-to-end script for running package)
        |__ models.py  (tensorflow estimator functions)
        |__ input_pipeline_dask.py  (handles input_pipeline)
        |__ utils
            |__ __init__.py
            |__ custom_utils_fn.py
            |__ optimizer_utils.py
            |__ metric_utils.py
```

# Prerequisites
Before you jump in, let’s cover some of the different tools you’ll be using to get your project running on ML Engine.

[Google Cloud Platform](https://cloud.google.com/) lets you build and host applications and websites, store data, and analyze data on Google's scalable infrastructure.

[Cloud ML Engine](https://cloud.google.com/ml-engine/) is a managed service that enables you to easily build machine learning models that work on any type of data, of any size.

[Google Cloud Storage](https://cloud.google.com/storage/) (GCS) is a unified object storage for developers and enterprises, from live data serving to data analytics/ML to data archiving.

[Cloud SDK](https://cloud.google.com/sdk/) is a command line tool which allows you to interact with Google Cloud products. In order to run this notebook, make sure that Cloud SDK is [installed](https://cloud.google.com/sdk/downloads) in the same environment as your Jupyter kernel.

# Steps to use
1. Make sure service are enabled
2. Modify metadata.py
3. Submit training job
