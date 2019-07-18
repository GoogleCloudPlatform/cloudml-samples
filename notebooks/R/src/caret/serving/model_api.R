library(plumber)

api <- plumb("model_prediction.R")
api$run(host = "0.0.0.0", port = 8080)