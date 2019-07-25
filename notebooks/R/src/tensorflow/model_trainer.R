library(tfestimators)
library(tfdatasets)
library(cloudml)
print("Libraries are imported.")

# Set your GCP project id
project_id <- "r-on-gcp"
# Set your GCS bucket
bucket <- "r-on-gcp" 

model_name <- 'tf_babyweight_estimator'
model_dir <- paste0('models/', model_name)
gcs_data_dir <- paste0("gs://", bucket, "/data")
gcs_model_dir <- paste0("gs://", bucket, "/models/", model_name)

header <- c(
    "weight_pounds", 
    "is_male", "mother_age", "mother_race", "plurality", "gestation_weeks", 
    "mother_married", "cigarette_use", "alcohol_use", 
    "key")

types <- c(
    "double", 
    "character", "double", "character", "double", "double", 
    "character", "character", "character", 
    "character")

target <- "weight_pounds"
key <- "key"
features <- setdiff(header, c(target, key))

create_estimator <-function(){
    print("Creating feature columns...")
    # numerical columns
    mother_age <- tf$feature_column$numeric_column("mother_age")
    plurality <- tf$feature_column$numeric_column('plurality')
    gestation_weeks <- tf$feature_column$numeric_column('gestation_weeks')


    # categorical columns
    is_male <- tf$feature_column$categorical_column_with_vocabulary_list("is_male", vocabulary_list = c("True", "False"))
    mother_race <- tf$feature_column$categorical_column_with_vocabulary_list(
        'mother_race', vocabulary_list = c('1', '2', '3', '4', '5', '6', '7', '8', '9', '18', '28', '38', '48', '58', '69', '78'))
    mother_married <- tf$feature_column$categorical_column_with_vocabulary_list('mother_married', c('True', 'False'))
    cigarette_use <- tf$feature_column$categorical_column_with_vocabulary_list('cigarette_use', c('True', 'False', 'None'))
    alcohol_use <- tf$feature_column$categorical_column_with_vocabulary_list('alcohol_use', c('True', 'False', 'None'))

    # extended feature columns
    cigarette_use_X_alcohol_use = tf$feature_column$crossed_column(c("cigarette_use", "alcohol_use"), 9)
    mother_race_embedded = tf$feature_column$embedding_column(mother_race, 3)
    mother_age_bucketized = tf$feature_column$bucketized_column(mother_age, boundaries=c(18, 22, 28, 32, 36, 40, 42, 45, 50))  
    mother_race_X_mother_age_bucketized = tf$feature_column$crossed_column(c(mother_age_bucketized, "mother_race"),  120)   
    mother_race_X_mother_age_bucketized_embedded = tf$feature_column$embedding_column(mother_race_X_mother_age_bucketized, 5)

    # wide and deep columns
    wide_columns <- feature_columns(
        is_male, mother_race, plurality, mother_married, cigarette_use, alcohol_use, cigarette_use_X_alcohol_use,
        mother_age_bucketized) 
    deep_columns <- feature_columns(
        mother_age, gestation_weeks, mother_race_embedded, mother_race_X_mother_age_bucketized_embedded)
    print("Feature columns created.")
    
    model <- dnn_linear_combined_regressor(
        model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_optimizer = "Adagrad",
        linear_optimizer = "Ftrl",
        dnn_hidden_units = c(64, 64),
        dnn_activation_fn = "relu", 
        dnn_dropout = 0.1,
    )
    
    return(model)

}

load_data <- function(){
    
    print("Downloading data files from GCS...")
    gsutil_exec("cp", "-r", gcs_data_dir, ".")
    print("Data files downloaded.")
    
    train_file <- "data/train_data.csv"
    eval_file <- "data/eval_data.csv"

    train_data <- read.table(train_file, col.names = header, sep=",", colClasses = types)
    eval_data <- read.table(eval_file, col.names = header, sep=",", colClasses = types)
    data <- list()
    data$train_data = train_data
    data$eval_data = eval_data
    return(data)
}

data_input_fn <- function(data, batch_size, num_epochs = 1, shuffle = FALSE) {
    input_fn(data, features = features, response = target, 
             batch_size = batch_size, shuffle = shuffle, num_epochs = num_epochs)
}

train_and_evaluate <- function(model, train_data, eval_data){
    batch_size = 64
    num_epochs = 2
    
    print("Training the model...")
    train(
        model, 
        input_fn = data_input_fn(train_data, batch_size = batch_size, num_epochs = num_epochs, shuffle = TRUE)
    )
    print("Model trained.")
    
    print("Evaluating the model...")
    eval_results <- evaluate(
        model, 
        input_fn = data_input_fn(eval_data, batch_size = batch_size)
    )
    print(paste0("Evaluation results: ", eval_results))
    
}

export_model <- function(model){
    print("Exporting the model...")
    feature_spec <- list()
    for (i in 1:length(header)) {
        column <- header[i]
        if (column %in% features) {

            default_value = 'NA'
            column_type <- types[i]

            if (column_type != 'character'){
                default_value = 0
            }

            default_tensor <- tf$constant(value = default_value, shape = shape(1, 1))
            feature_spec[[column]] <- tf$placeholder_with_default(
                input = default_tensor, shape = shape(NULL, 1))
        }
    
    }  
            
    serving_input_receiver_fn <- tf$estimator$export$build_raw_serving_input_receiver_fn(feature_spec)
    saved_model_dir = paste0(model_dir, '/export')
    export_savedmodel(model, saved_model_dir, serving_input_receiver_fn = serving_input_receiver_fn)
    print("Model exported.")
    
    print("Uploading the saved model to GCS...")
    gsutil_exec("cp", "-r", paste0(saved_model_dir,"/*"), gcs_model_dir)
    print("Saved model uploaded.")
}


main <-function(args){
  
    print("Trainer started...")
    data <- load_data()
    model <- create_estimator()
    train_and_evaluate(model, data$train_data, data$eval_data)
    export_model(model)
    print("Trained finished.")
    

}

main()