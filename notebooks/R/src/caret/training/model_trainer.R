library(caret)
print("Libraries are imported.")

# Set the project id
project_id <- "r-on-gcp"
# Set yout GCS bucket
bucket <- "r_on_gcp" 

model_name <- 'caret_babyweight_estimator'
model_dir <- paste0('models/', model_name)
gcs_data_dir <- paste0("gs://", bucket, "/data")
gcs_model_dir <- paste0("gs://", bucket, "/models/", model_name, "/")


header <- c(
    "weight_pounds", 
    "is_male", "mother_age", "mother_race", "plurality", "gestation_weeks", 
    "mother_married", "cigarette_use", "alcohol_use", 
    "key")

target <- "weight_pounds"
key <- "key"
features <- setdiff(header, c(target, key))


load_data <- function(){
    print("Downloading data files from GCS...")
    command <- paste("gsutil cp -r", gcs_data_dir, ".")
    system(command)
    print("Data files downloaded.")
    
    train_file <- "data/train_data.csv"
    eval_file <- "data/eval_data.csv"

    train_data <- read.table(train_file, col.names = header, sep=",")
    eval_data <- read.table(eval_file, col.names = header, sep=",")
    data <- list()
    data$train_data = train_data
    data$eval_data = eval_data
    return(data)
}

train_model <-function(train_data){
    
    trainControl <- trainControl(method = 'boot', number = 10)
        hyper_parameters <- expand.grid(
            nrounds = 100,
            max_depth = 6,
            eta = 0.3,
            gamma = 0,
            colsample_bytree = 1,
            min_child_weight = 1,
            subsample = 1
    )

    print('Training the model...')

    model <- train(
        y=train_data$weight_pounds, 
        x=train_data[, features], 
        preProc = c("center", "scale"),
        method='xgbTree', 
        trControl=trainControl,
        tuneGrid=hyper_parameters
    )

    print('Model is trained.')
    return(model)

}


export_model <- function(model){
    print("Exporting the model...")
    model_dir <- "models"
    model_name <- "caret_babyweight_estimator"
    dir.create(model_dir, showWarnings = FALSE)
    dir.create(file.path(model_dir, model_name), showWarnings = FALSE)
    saved_model_dir <- file.path(model_dir, model_name, "trained_model.rds")
    saveRDS(model, saved_model_dir)
    print("Model exported.")
    
    print("Uploading the saved model to GCS...")
    command <- paste("gsutil cp", saved_model_dir, gcs_model_dir)
    system(command)
    print("Saved model uploaded.")
}


main <-function(args){
  
    print("Trainer started...")
    data <- load_data()
    model <- train_model(data$train_data)
    print(model)
    export_model(model)
    print("Trained finished.")
    
#   if (length(args)!=3) {
#     stop("You need to specify projectid, bucket, and where to save the model in gcs.\n", 
#          call.=FALSE)
#     return(NULL)
#   }
  
#  projectid = args[1]
#   bucket = args[2]
#   gcs_model_location = args[3]
  
#   print(projectid)
#   print(bucket)
#   print(gcs_model_location)
    
  


}

#args = commandArgs(trailingOnly=TRUE)
main()