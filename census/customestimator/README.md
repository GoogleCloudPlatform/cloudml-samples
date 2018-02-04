We use a custom estimator to construct a census model. The difference between 
this model and the other two (tensorflow-core and canned estimator) is that
a numeric 'key' is required in the input. The key will be passed through from
the input to output. It is useful for the scenario of CMLE batch prediction where
the output predictions are out of order. 


* In the CSV input, specify a number in the first column in the input.
* In the Example Proto input, specify a number in the numeric feature called 'key' in the input.
* In the JSON input, specify a number in the input JSON object with the key of 'key'.
