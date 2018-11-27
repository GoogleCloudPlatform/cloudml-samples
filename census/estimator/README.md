
Dataflow:
The current code uses the test set to control overfitting or to do hyper parameter tuning, while we typically recommend to use an evaluation set for this.
Therefore, the Dataflow pipeline splits the train set into train and validation sets. While this is a very basic use of Dataflow, we hope that this simple example illustrates the typical coding paradigm of Dataflow.

--project_name mvm-implementation