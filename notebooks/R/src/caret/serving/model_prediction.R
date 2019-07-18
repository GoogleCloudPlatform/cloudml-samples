# Set the project id
project_id <- "r-on-gcp"

# Set yout GCS bucket
bucket <- "r_on_gcp"

model_name <- 'caret_babyweight_estimator'
model_dir <- paste0('models/', model_name)
gcs_model_dir <- paste0("gs://", bucket, "/models/", model_name)

print("Downloading model file from GCS...")
command <- paste("gsutil cp -r", gcs_model_dir, ".")
system(command)
print("model files downloaded.")

print("Loading model ...")
model <- readRDS(paste0(model_name,'/trained_model.rds'))
print("Model is loaded.")

#* @post /estimate
estimate_babyweights <- function(req){
  library("rjson")
  instances_json <- req$postBody
  instances <- jsonlite::fromJSON(instances_json)
  df_instances <- data.frame(instances)
  # fix data types
  boolean_columns <- c("is_male", "mother_married", "cigarette_use", "alcohol_use")
  for(col in boolean_columns){
    df_instances[[col]] <- as.logical(df_instances[[col]])
  }

  estimates <- predict(model, df_instances)
  return(estimates)

}





