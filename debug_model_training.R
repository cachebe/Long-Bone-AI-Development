# Quick diagnostic script to identify the model training failure

library(glmnet)
library(xgboost)
library(caret)

# Load data
model1_data <- readRDS("poc_model1_data.rds")
model2_data <- readRDS("poc_model_data.rds")

cat("==============================================================================\n")
cat("DIAGNOSTIC: Identifying Model Training Failure\n")
cat("==============================================================================\n\n")

# Check the structure of the outcome variable
cat("1. Checking outcome variable structure:\n")
cat("   Class of Nonunion_Label: ", class(model2_data$Nonunion_Label), "\n")
cat("   Type: ", typeof(model2_data$Nonunion_Label), "\n")
cat("   Unique values: ", paste(unique(model2_data$Nonunion_Label), collapse=", "), "\n")
cat("   Is factor? ", is.factor(model2_data$Nonunion_Label), "\n")
cat("   Is numeric? ", is.numeric(model2_data$Nonunion_Label), "\n\n")

# Try the prepare_data function
prepare_data <- function(data) {
  X <- as.matrix(data[, -1])
  y <- data$Nonunion_Label  # Current version
  list(X = X, y = y)
}

data2 <- prepare_data(model2_data)

cat("2. After prepare_data:\n")
cat("   Class of y: ", class(data2$y), "\n")
cat("   Type: ", typeof(data2$y), "\n")
cat("   Unique values: ", paste(unique(data2$y), collapse=", "), "\n")
cat("   Is numeric? ", is.numeric(data2$y), "\n\n")

# Create a simple train/test split
set.seed(42)
folds <- createFolds(data2$y, k = 5, list = TRUE, returnTrain = FALSE)
test_idx <- folds[[1]]
train_idx <- setdiff(1:nrow(data2$X), test_idx)

X_train <- data2$X[train_idx, , drop = FALSE]
y_train <- data2$y[train_idx]

cat("3. Training data:\n")
cat("   N samples: ", length(y_train), "\n")
cat("   N features: ", ncol(X_train), "\n")
cat("   Class of y_train: ", class(y_train), "\n")
cat("   Unique values in y_train: ", paste(unique(y_train), collapse=", "), "\n")
cat("   Table of y_train:\n")
print(table(y_train))
cat("\n")

# Calculate sample weights
n_nonunion <- sum(model2_data$Nonunion_Label == 1)
n_union <- sum(model2_data$Nonunion_Label == 0)
class_weights <- c(
  "Union" = nrow(model2_data) / (2 * n_union),
  "Nonunion" = nrow(model2_data) / (2 * n_nonunion)
)

sample_weights <- ifelse(y_train == 1,
                         class_weights["Nonunion"],
                         class_weights["Union"])

cat("4. Sample weights:\n")
cat("   Length: ", length(sample_weights), "\n")
cat("   Range: ", min(sample_weights), " to ", max(sample_weights), "\n\n")

# Try to train a simple glmnet model
cat("5. Attempting to train cv.glmnet model:\n")
cat("   -------------------------------------------\n")

result <- tryCatch({
  cv_fit <- cv.glmnet(
    x = X_train,
    y = y_train,
    family = "binomial",
    alpha = 0.5,
    weights = sample_weights,
    type.measure = "auc",
    nfolds = 3,
    standardize = TRUE,
    maxit = 100000
  )
  cat("   ✓ SUCCESS! Model trained.\n")
  cat("   Lambda.1se: ", cv_fit$lambda.1se, "\n")
  cat("   Max AUC: ", max(cv_fit$cvm), "\n")
  cv_fit
}, error = function(e) {
  cat("   ✗ ERROR caught:\n")
  cat("   Message: ", e$message, "\n")
  cat("   Call: ", deparse(e$call), "\n")
  NULL
})

cat("\n")

# Try XGBoost
cat("6. Attempting to train XGBoost model:\n")
cat("   -------------------------------------------\n")

# Check if y_train needs conversion
if (!is.numeric(y_train)) {
  cat("   WARNING: y_train is not numeric, converting...\n")
  y_train_numeric <- as.numeric(as.character(y_train))
} else {
  y_train_numeric <- y_train
}

cat("   y_train_numeric unique values: ", paste(unique(y_train_numeric), collapse=", "), "\n")

dtrain <- xgb.DMatrix(data = X_train, label = y_train_numeric, weight = sample_weights)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 2,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5
)

result_xgb <- tryCatch({
  cv_result <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 50,
    nfold = 3,
    early_stopping_rounds = 10,
    verbose = 0,
    stratified = TRUE
  )
  cat("   ✓ SUCCESS! XGBoost CV completed.\n")
  cat("   Best iteration: ", cv_result$best_iteration, "\n")
  cat("   Best AUC: ", max(cv_result$evaluation_log$test_auc_mean, na.rm=TRUE), "\n")
  cv_result
}, error = function(e) {
  cat("   ✗ ERROR caught:\n")
  cat("   Message: ", e$message, "\n")
  cat("   Call: ", deparse(e$call), "\n")
  NULL
})

cat("\n==============================================================================\n")
cat("DIAGNOSIS COMPLETE\n")
cat("==============================================================================\n")
