# ==============================================================================
# Tibial Fracture Nonunion Prediction Model Comparison (v2 - MISSING VALUES FIXED)
# ==============================================================================
# Purpose: Compare two prediction models using nested cross-validation:
#   - Model 1: Baseline (Clinical + PROMIS features)
#   - Model 2: Full (Clinical + PROMIS + RUST features)
#
# FIXES in v2:
#   - Proper factor-to-numeric conversion
#   - Missing value imputation (median for continuous, mode for binary)
#   - Better error reporting
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Load Required Libraries
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("TIBIAL NONUNION PREDICTION MODEL COMPARISON (v2)\n")
cat("==============================================================================\n\n")

cat("Loading required libraries...\n")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(caret)
  library(glmnet)
  library(xgboost)
  library(pROC)
  library(ggplot2)
  library(gridExtra)
  library(boot)
  library(PredictABEL)
})

options(scipen = 999)
set.seed(42)

cat("  ✓ All libraries loaded successfully\n\n")

# ------------------------------------------------------------------------------
# 2. Load and Prepare Data with Missing Value Handling
# ------------------------------------------------------------------------------
cat("Loading datasets...\n")

model1_data <- readRDS("poc_model1_data.rds")
model2_data <- readRDS("poc_model_data.rds")

cat("  ✓ Model 1 (Baseline): ", nrow(model1_data), " patients, ",
    ncol(model1_data) - 1, " features\n", sep = "")
cat("  ✓ Model 2 (Full):     ", nrow(model2_data), " patients, ",
    ncol(model2_data) - 1, " features\n\n", sep = "")

# ==============================================================================
# FIX: Function to impute missing values
# ==============================================================================
impute_missing <- function(X) {
  # X is a matrix
  if (sum(is.na(X)) == 0) {
    return(X)
  }

  cat("    Imputing missing values...\n")
  cat("    Total NAs before imputation: ", sum(is.na(X)), "\n")

  X_imputed <- X
  for (j in 1:ncol(X)) {
    if (any(is.na(X[, j]))) {
      # Check if binary (only 0 and 1)
      unique_vals <- unique(X[!is.na(X[, j]), j])
      if (all(unique_vals %in% c(0, 1))) {
        # Binary: use mode (most frequent value)
        mode_val <- as.numeric(names(sort(table(X[!is.na(X[, j]), j]), decreasing=TRUE)[1]))
        X_imputed[is.na(X[, j]), j] <- mode_val
      } else {
        # Continuous: use median
        median_val <- median(X[!is.na(X[, j]), j], na.rm = TRUE)
        X_imputed[is.na(X[, j]), j] <- median_val
      }
    }
  }

  cat("    Total NAs after imputation: ", sum(is.na(X_imputed)), "\n")
  return(X_imputed)
}

# ==============================================================================
# FIX: Properly convert outcome variable and handle missing values
# ==============================================================================
prepare_data <- function(data) {
  # Separate features and outcome
  X <- as.matrix(data[, -1])

  # Ensure y is numeric 0/1
  if (is.factor(data$Nonunion_Label)) {
    y <- as.numeric(data$Nonunion_Label) - 1
  } else if (is.character(data$Nonunion_Label)) {
    y <- ifelse(data$Nonunion_Label %in% c("Nonunion", "1"), 1, 0)
  } else {
    y <- as.numeric(data$Nonunion_Label)
  }

  # Verify y is 0/1
  if (!all(y %in% c(0, 1))) {
    stop("Outcome variable must be 0/1. Found values: ", paste(unique(y), collapse=", "))
  }

  # Handle missing values in X
  if (sum(is.na(X)) > 0) {
    X <- impute_missing(X)
  }

  # Final check
  if (sum(is.na(X)) > 0) {
    stop("ERROR: Missing values remain after imputation!")
  }
  if (any(is.na(y))) {
    stop("ERROR: Missing values in outcome variable!")
  }

  list(X = X, y = y)
}

cat("Preparing data...\n")
data1 <- prepare_data(model1_data)
data2 <- prepare_data(model2_data)

cat("\nData preparation complete:\n")
cat("  - Model 1: X is ", nrow(data1$X), " x ", ncol(data1$X), ", y length = ", length(data1$y), "\n", sep="")
cat("  - Model 2: X is ", nrow(data2$X), " x ", ncol(data2$X), ", y length = ", length(data2$y), "\n", sep="")
cat("  - Missing values in Model 1 X: ", sum(is.na(data1$X)), "\n", sep="")
cat("  - Missing values in Model 2 X: ", sum(is.na(data2$X)), "\n", sep="")
cat("  - y is numeric: ", is.numeric(data2$y), "\n", sep="")
cat("  - y unique values: ", paste(unique(data2$y), collapse=", "), "\n\n", sep="")

# Class distribution
n_nonunion <- sum(data2$y == 1)
n_union <- sum(data2$y == 0)
imbalance_ratio <- n_union / n_nonunion

cat("Class distribution:\n")
cat("  - Nonunion (positive class): ", n_nonunion, " (",
    round(100 * n_nonunion / length(data2$y), 1), "%)\n", sep = "")
cat("  - Union (negative class):    ", n_union, " (",
    round(100 * n_union / length(data2$y), 1), "%)\n", sep = "")
cat("  - Imbalance ratio: ", round(imbalance_ratio, 2), ":1\n\n", sep = "")

# Class weights
class_weights <- c(
  "Union" = length(data2$y) / (2 * n_union),
  "Nonunion" = length(data2$y) / (2 * n_nonunion)
)

cat("Class weights:\n")
cat("  - Union weight:    ", round(class_weights["Union"], 3), "\n", sep = "")
cat("  - Nonunion weight: ", round(class_weights["Nonunion"], 3), "\n\n", sep = "")

# ------------------------------------------------------------------------------
# 3. Model Training Functions
# ------------------------------------------------------------------------------
cat("Setting up nested cross-validation...\n")

n_outer_folds <- 5
n_inner_folds <- 5
n_repetitions <- 10

cat("  - Outer folds: ", n_outer_folds, "\n", sep = "")
cat("  - Inner folds: ", n_inner_folds, "\n", sep = "")
cat("  - Repetitions: ", n_repetitions, "\n\n", sep = "")

create_stratified_folds <- function(y, k = 5, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  createFolds(y, k = k, list = TRUE, returnTrain = FALSE)
}

train_elastic_net <- function(X_train, y_train, X_val = NULL, y_val = NULL,
                              class_weights, tune = TRUE) {

  # Verify no missing values
  if (any(is.na(X_train)) || any(is.na(y_train))) {
    cat("    ERROR: Missing values in training data!\n")
    return(list(model = NULL, type = "glmnet"))
  }

  sample_weights <- ifelse(y_train == 1,
                           class_weights["Nonunion"],
                           class_weights["Union"])

  if (tune) {
    alpha_grid <- seq(0, 1, by = 0.2)
    best_auc <- 0
    best_alpha <- 1
    best_model <- NULL

    for (alpha in alpha_grid) {
      cv_fit <- tryCatch({
        cv.glmnet(
          x = X_train,
          y = y_train,
          family = "binomial",
          alpha = alpha,
          weights = sample_weights,
          type.measure = "auc",
          nfolds = 3,
          standardize = TRUE,
          maxit = 100000
        )
      }, error = function(e) {
        cat("    ERROR in cv.glmnet (alpha=", alpha, "): ", e$message, "\n", sep="")
        NULL
      })

      if (!is.null(cv_fit)) {
        current_auc <- max(cv_fit$cvm)
        if (current_auc > best_auc) {
          best_auc <- current_auc
          best_alpha <- alpha
          best_model <- cv_fit
        }
      }
    }

    list(
      model = best_model,
      alpha = best_alpha,
      lambda = if(!is.null(best_model)) best_model$lambda.1se else NULL,
      type = "glmnet"
    )

  } else {
    cv_fit <- tryCatch({
      cv.glmnet(
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
    }, error = function(e) {
      cat("    ERROR in cv.glmnet: ", e$message, "\n", sep="")
      NULL
    })

    if (is.null(cv_fit)) {
      return(list(model = NULL, type = "glmnet"))
    }

    list(
      model = cv_fit,
      alpha = 0.5,
      lambda = cv_fit$lambda.1se,
      type = "glmnet"
    )
  }
}

train_xgboost <- function(X_train, y_train, X_val = NULL, y_val = NULL,
                          class_weights, tune = TRUE) {

  # Verify no missing values
  if (any(is.na(X_train)) || any(is.na(y_train))) {
    cat("    ERROR: Missing values in training data!\n")
    return(list(model = NULL, type = "xgboost"))
  }

  # Verify y is 0/1
  if (!all(y_train %in% c(0, 1))) {
    cat("    ERROR: y_train must be 0/1, found: ", paste(unique(y_train), collapse=", "), "\n", sep="")
    return(list(model = NULL, type = "xgboost"))
  }

  sample_weights <- ifelse(y_train == 1,
                           class_weights["Nonunion"],
                           class_weights["Union"])

  if(length(unique(y_train)) < 2) {
    cat("    WARNING: XGBoost - only one class in training data\n")
    return(list(model = NULL, type = "xgboost"))
  }

  dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = sample_weights,
                        missing = NA)  # Explicitly handle NA (though should be none)

  if (tune) {
    param_grid <- expand.grid(
      max_depth = c(2, 3),
      eta = c(0.1, 0.3),
      stringsAsFactors = FALSE
    )

    best_auc <- 0
    best_params <- NULL
    best_model <- NULL

    for (i in 1:nrow(param_grid)) {
      params <- list(
        objective = "binary:logistic",
        eval_metric = "auc",
        max_depth = param_grid$max_depth[i],
        eta = param_grid$eta[i],
        subsample = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5
      )

      cv_result <- tryCatch({
        xgb.cv(
          params = params,
          data = dtrain,
          nrounds = 50,
          nfold = 3,
          early_stopping_rounds = 10,
          verbose = 0,
          stratified = TRUE
        )
      }, error = function(e) {
        cat("    ERROR in xgb.cv: ", e$message, "\n", sep="")
        NULL
      })

      if (!is.null(cv_result)) {
        best_iter <- cv_result$best_iteration
        current_auc <- max(cv_result$evaluation_log$test_auc_mean, na.rm = TRUE)

        if (!is.na(current_auc) && current_auc > best_auc) {
          best_auc <- current_auc
          best_params <- params
          best_params$nrounds <- best_iter

          best_model <- xgb.train(
            params = params,
            data = dtrain,
            nrounds = best_iter,
            verbose = 0
          )
        }
      }
    }

    list(
      model = best_model,
      params = best_params,
      type = "xgboost"
    )

  } else {
    params <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = 2,
      eta = 0.1,
      subsample = 0.8,
      colsample_bytree = 0.8,
      min_child_weight = 5
    )

    cv_result <- tryCatch({
      xgb.cv(
        params = params,
        data = dtrain,
        nrounds = 50,
        nfold = 3,
        early_stopping_rounds = 10,
        verbose = 0,
        stratified = TRUE
      )
    }, error = function(e) {
      cat("    ERROR in xgb.cv: ", e$message, "\n", sep="")
      NULL
    })

    if (is.null(cv_result)) {
      return(list(model = NULL, type = "xgboost"))
    }

    best_iter <- cv_result$best_iteration

    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = best_iter,
      verbose = 0
    )

    list(
      model = model,
      params = params,
      type = "xgboost"
    )
  }
}

make_predictions <- function(model_object, X_new) {
  if (model_object$type == "glmnet") {
    pred <- predict(model_object$model, newx = X_new, s = "lambda.1se", type = "response")
    as.numeric(pred)
  } else if (model_object$type == "xgboost") {
    dtest <- xgb.DMatrix(data = X_new, missing = NA)
    predict(model_object$model, dtest)
  }
}

cat("  ✓ Model training functions defined\n\n")

# ------------------------------------------------------------------------------
# 4. Nested Cross-Validation Execution
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("RUNNING NESTED CROSS-VALIDATION\n")
cat("==============================================================================\n\n")

cv_results <- list(
  model1_elastic = list(),
  model2_elastic = list(),
  model1_xgboost = list(),
  model2_xgboost = list()
)

all_predictions <- data.frame(
  repetition = integer(),
  fold = integer(),
  index = integer(),
  true_label = character(),
  model1_elastic_pred = numeric(),
  model2_elastic_pred = numeric(),
  model1_xgb_pred = numeric(),
  model2_xgb_pred = numeric(),
  stringsAsFactors = FALSE
)

start_time <- Sys.time()

for (rep in 1:n_repetitions) {
  cat("Repetition ", rep, "/", n_repetitions, "\n", sep = "")

  folds <- create_stratified_folds(data2$y, k = n_outer_folds, seed = rep * 100)

  for (fold in 1:n_outer_folds) {
    cat("  Fold ", fold, "/", n_outer_folds, "... ", sep = "")

    test_idx <- folds[[fold]]
    train_idx <- setdiff(1:nrow(data2$X), test_idx)

    if(length(unique(data2$y[train_idx])) < 2 || length(unique(data2$y[test_idx])) < 2) {
      cat("SKIPPED - insufficient class diversity\n")
      next
    }

    # Get data splits
    X1_train <- data1$X[train_idx, , drop = FALSE]
    X1_test <- data1$X[test_idx, , drop = FALSE]
    y_train <- data1$y[train_idx]
    y_test <- data1$y[test_idx]

    X2_train <- data2$X[train_idx, , drop = FALSE]
    X2_test <- data2$X[test_idx, , drop = FALSE]

    # Train models
    model1_en <- train_elastic_net(X1_train, y_train, class_weights = class_weights, tune = TRUE)
    model2_en <- train_elastic_net(X2_train, y_train, class_weights = class_weights, tune = TRUE)
    model1_xgb <- train_xgboost(X1_train, y_train, class_weights = class_weights, tune = TRUE)
    model2_xgb <- train_xgboost(X2_train, y_train, class_weights = class_weights, tune = TRUE)

    if (is.null(model1_en$model) || is.null(model2_en$model) ||
        is.null(model1_xgb$model) || is.null(model2_xgb$model)) {
      cat("FAILED - model training returned NULL\n")
      next
    }

    # Make predictions
    pred1_en <- make_predictions(model1_en, X1_test)
    pred2_en <- make_predictions(model2_en, X2_test)
    pred1_xgb <- make_predictions(model1_xgb, X1_test)
    pred2_xgb <- make_predictions(model2_xgb, X2_test)

    # For ROC, convert y_test to factor
    y_test_factor <- factor(y_test, levels = c(0, 1), labels = c("Union", "Nonunion"))

    # Calculate AUCs
    roc1_en <- roc(y_test_factor, pred1_en, levels = c("Union", "Nonunion"), direction = "<", quiet = TRUE)
    roc2_en <- roc(y_test_factor, pred2_en, levels = c("Union", "Nonunion"), direction = "<", quiet = TRUE)
    roc1_xgb <- roc(y_test_factor, pred1_xgb, levels = c("Union", "Nonunion"), direction = "<", quiet = TRUE)
    roc2_xgb <- roc(y_test_factor, pred2_xgb, levels = c("Union", "Nonunion"), direction = "<", quiet = TRUE)

    # Store results
    cv_results$model1_elastic[[length(cv_results$model1_elastic) + 1]] <- list(
      repetition = rep, fold = fold, auc = as.numeric(auc(roc1_en)),
      roc = roc1_en, predictions = pred1_en, true_labels = y_test
    )
    cv_results$model2_elastic[[length(cv_results$model2_elastic) + 1]] <- list(
      repetition = rep, fold = fold, auc = as.numeric(auc(roc2_en)),
      roc = roc2_en, predictions = pred2_en, true_labels = y_test
    )
    cv_results$model1_xgboost[[length(cv_results$model1_xgboost) + 1]] <- list(
      repetition = rep, fold = fold, auc = as.numeric(auc(roc1_xgb)),
      roc = roc1_xgb, predictions = pred1_xgb, true_labels = y_test
    )
    cv_results$model2_xgboost[[length(cv_results$model2_xgboost) + 1]] <- list(
      repetition = rep, fold = fold, auc = as.numeric(auc(roc2_xgb)),
      roc = roc2_xgb, predictions = pred2_xgb, true_labels = y_test
    )

    # Store predictions
    fold_predictions <- data.frame(
      repetition = rep, fold = fold, index = test_idx,
      true_label = as.character(y_test),
      model1_elastic_pred = pred1_en, model2_elastic_pred = pred2_en,
      model1_xgb_pred = pred1_xgb, model2_xgb_pred = pred2_xgb,
      stringsAsFactors = FALSE
    )
    all_predictions <- rbind(all_predictions, fold_predictions)

    cat("AUC_M1_EN=", round(as.numeric(auc(roc1_en)), 3),
        " AUC_M2_EN=", round(as.numeric(auc(roc2_en)), 3), "\n", sep = "")
  }
  cat("\n")
}

end_time <- Sys.time()
elapsed_time <- difftime(end_time, start_time, units = "mins")

cat("✓ Nested CV completed in ", round(elapsed_time, 1), " minutes\n\n", sep = "")
cat("✓ Total successful folds: ", nrow(all_predictions) / length(unique(all_predictions$index)), "\n\n", sep="")

# Save intermediate results
saveRDS(list(cv_results = cv_results, all_predictions = all_predictions),
        "cv_results_intermediate.rds")
cat("✓ Intermediate results saved to: cv_results_intermediate.rds\n\n")

cat("==============================================================================\n")
cat("CROSS-VALIDATION COMPLETE!\n")
cat("==============================================================================\n\n")

cat("Successfully completed ", length(cv_results$model1_elastic), " folds out of ",
    n_repetitions * n_outer_folds, " total folds\n\n", sep="")

cat("Next steps:\n")
cat("  1. Check cv_results_intermediate.rds for results\n")
cat("  2. Run statistical analysis (DeLong, Bootstrap, NRI/IDI)\n")
cat("  3. Generate plots and summary tables\n\n")
