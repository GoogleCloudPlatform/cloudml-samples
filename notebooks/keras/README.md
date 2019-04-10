# Cascade (HD-CNN Model Derivative)

This notebook demonstrates building a hierarchical image classifier based on a HD-CNN derivative which uses cascading classifiers to predict the class of a label from a coarse to finer classes.

In this demonstration, we have two classes in the hierarchy: fruits and varieties of fruit. The model will first predict the coarse class (type of fruit) and then within that class of fruit, the variety. For example, if given an image of Apple Granny Smith, it would first predict 'Apple' (fruit) and then predict the 'Apple Granny Smith'.

This derivative of the HD-CNN is designed to demonstrate both the methodology of hierarchical classification, as well as design improvements not available at the time (2014) when the model was first published Zhicheng Yan.

# General Approach

Our HD-CNN deriative archirecture consists of:

1. An stem convolutional block.
    - The output from the stem convolutional head is shared with the coarse and finer classifiers 
    (referred to as the shared layers in the paper).
2. A coarse classifier.
    - A Convolution and Dense layers for classifying the coarse level class. 
3. A set of finer classifiers, one per coarse level class.
    - A Convolution and Dense layers per coarse level class for classifying the corresponding finer 
    level class.
4. A conditional execution step for predicting a specific finer classifier based on the output of the 
   coarse classifier.
    - The coarse level classifier is predicted.
    - The index of the prediction is used to select a finer classifier.
    - An im-memory copy of the shared bottleneck layer (i.e., last convolution layer in stem) is passed as the
      input to the finer level classifier.


## Execution Instructions

This notebook requires 17GB of memory. It will not run on a Standard TF JaaS instance (15GB). You will need to select an instance with memory > 17GB.

Some of the cells in the notebook display images. The images will not appear until the cell for copying the training data/misc from GCS into the JaaS instance is executed.
