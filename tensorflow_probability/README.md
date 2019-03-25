# TensorFlow Probability Examples on Cloud Machine Learning Engine.


## Overview

This directory contains examples imported from the [tensorflow/probability repository](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples) with boilerplate pieces added to be runnable on [Cloud Machine Learning Engine](http://cloud.google.com/ml-engine).


## Usage

1. Clone this repository:

```
git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git
```

1. Navigate to one of the examples, for example the `grammar_vae` example:

```
cd cloudml-samples/tensorflow_probability/grammar_vae
```

1. Modify the `BUCKET` variable in `submit.sh` to point to a Google Cloud Storage bucket you have write access to.

1. Submit a training job:

```
bash submit.sh
```
