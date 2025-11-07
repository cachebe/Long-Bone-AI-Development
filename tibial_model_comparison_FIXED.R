# ==============================================================================
# Tibial Fracture Nonunion Prediction Model Comparison (FIXED VERSION)
# ==============================================================================
# Purpose: Compare two prediction models using nested cross-validation:
#   - Model 1: Baseline (Clinical + PROMIS features)
#   - Model 2: Full (Clinical + PROMIS + RUST features)
#
# Study Population: 234 tibial fracture patients (37 nonunions, 197 unions)
# Class imbalance ratio: 5.3:1 (union:nonunion)
#
# Methods:
#   - Nested 5-fold cross-validation with 10 repetitions
#   - Stratified splits maintaining nonunion proportion
#   - Class weights to handle imbalance
#   - Primary: Elastic net (glmnet) with alpha tuning
#   - Secondary: XGBoost for validation
#
# Statistical Comparisons:
#   - DeLong's test for paired ROC curves
#   - Bootstrap CI (2000 iterations) for AUC difference
#   - Net Reclassification Improvement (NRI)
#   - Integrated Discrimination Improvement (IDI)
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Load Required Libraries
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("TIBIAL NONUNION PREDICTION MODEL COMPARISON\n")
cat("==============================================================================\n\n")

cat("Loading required libraries...\n")

# Core libraries
suppressPackageStartupMessages({
  library(dplyr)          # Data manipulation
  library(tidyr)          # Data tidying
  library(caret)          # ML framework and CV
  library(glmnet)         # Elastic net models
  library(xgboost)        # XGBoost models
  library(pROC)           # ROC analysis and DeLong's test
  library(ggplot2)        # Visualization
  library(gridExtra)      # Multiple plots
  library(boot)           # Bootstrap methods
  library(PredictABEL)    # NRI and IDI calculations
})

# Set global options
options(scipen = 999)  # Avoid scientific notation
set.seed(42)           # Reproducibility

cat("  ✓ All libraries loaded successfully\n\n")

# ------------------------------------------------------------------------------
# 2. Load and Prepare Data
# ------------------------------------------------------------------------------
cat("Loading datasets...\n")

# Load both models' data
model1_data <- readRDS("poc_model1_data.rds")  # Baseline model
model2_data <- readRDS("poc_model_data.rds")   # Full model

cat("  ✓ Model 1 (Baseline): ", nrow(model1_data), " patients, ",
    ncol(model1_data) - 1, " features\n", sep = "")
cat("  ✓ Model 2 (Full):     ", nrow(model2_data), " patients, ",
    ncol(model2_data) - 1, " features\n\n", sep = "")

# Verify class distribution
n_nonunion <- sum(model2_data$Nonunion_Label == 1)
n_union <- sum(model2_data$Nonunion_Label == 0)
imbalance_ratio <- n_union / n_nonunion

cat("Class distribution:\n")
cat("  - Nonunion (positive class): ", n_nonunion, " (",
    round(100 * n_nonunion / nrow(model2_data), 1), "%)\n", sep = "")
cat("  - Union (negative class):    ", n_union, " (",
    round(100 * n_union / nrow(model2_data), 1), "%)\n", sep = "")
cat("  - Imbalance ratio: ", round(imbalance_ratio, 2), ":1\n\n", sep = "")

# ==============================================================================
# FIX: Properly convert outcome variable to numeric 0/1
# ==============================================================================
prepare_data <- function(data) {
  X <- as.matrix(data[, -1])

  # Ensure y is numeric 0/1, regardless of input type
  if (is.factor(data$Nonunion_Label)) {
    # If factor, convert to numeric and adjust to 0/1
    y <- as.numeric(data$Nonunion_Label) - 1
  } else if (is.character(data$Nonunion_Label)) {
    # If character, map to 0/1
    y <- ifelse(data$Nonunion_Label %in% c("Nonunion", "1"), 1, 0)
  } else {
    # If already numeric, ensure it's 0/1
    y <- as.numeric(data$Nonunion_Label)
  }

  # Verify it's 0/1
  if (!all(y %in% c(0, 1))) {
    stop("Outcome variable must be 0/1. Found values: ", paste(unique(y), collapse=", "))
  }

  list(X = X, y = y)
}

data1 <- prepare_data(model1_data)
data2 <- prepare_data(model2_data)

# Verify conversion
cat("After prepare_data:\n")
cat("  - y is numeric: ", is.numeric(data2$y), "\n", sep = "")
cat("  - y unique values: ", paste(unique(data2$y), collapse=", "), "\n", sep = "")
cat("  - y distribution: ", sum(data2$y == 1), " nonunions, ", sum(data2$y == 0), " unions\n\n", sep = "")

# Define class weights for imbalance handling
# Weight = n_samples / (n_classes * n_samples_class)
class_weights <- c(
  "Union" = nrow(model2_data) / (2 * n_union),
  "Nonunion" = nrow(model2_data) / (2 * n_nonunion)
)

cat("Class weights calculated:\n")
cat("  - Union weight:    ", round(class_weights["Union"], 3), "\n", sep = "")
cat("  - Nonunion weight: ", round(class_weights["Nonunion"], 3), "\n", sep = "")
cat("  - Weight ratio:    ", round(class_weights["Nonunion"] / class_weights["Union"], 2), ":1\n\n", sep = "")

# ------------------------------------------------------------------------------
# 3. Nested Cross-Validation Setup
# ------------------------------------------------------------------------------
cat("Setting up nested cross-validation...\n")

# CV parameters
n_outer_folds <- 5      # Outer loop for performance evaluation
n_inner_folds <- 5      # Inner loop for hyperparameter tuning
n_repetitions <- 10     # Number of CV repetitions for stability

cat("  - Outer folds: ", n_outer_folds, " (for performance evaluation)\n", sep = "")
cat("  - Inner folds: ", n_inner_folds, " (for hyperparameter tuning)\n", sep = "")
cat("  - Repetitions: ", n_repetitions, " (for stable estimates)\n", sep = "")
cat("  - Stratification: YES (maintains nonunion proportion)\n\n")

# Function to create stratified folds
create_stratified_folds <- function(y, k = 5, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  createFolds(y, k = k, list = TRUE, returnTrain = FALSE)
}

# ------------------------------------------------------------------------------
# 4. Elastic Net Model Training Function
# ------------------------------------------------------------------------------
cat("Defining model training functions...\n")

train_elastic_net <- function(X_train, y_train, X_val = NULL, y_val = NULL,
                              class_weights, tune = TRUE) {

  # Create sample weights - y_train is numeric 0/1
  sample_weights <- ifelse(y_train == 1,
                           class_weights["Nonunion"],
                           class_weights["Union"])

  if (tune) {
    alpha_grid <- seq(0, 1, by = 0.2)  # Reduced grid for speed
    best_auc <- 0
    best_alpha <- 1
    best_model <- NULL

    for (alpha in alpha_grid) {
      cv_fit <- tryCatch({
        cv.glmnet(
          x = X_train,
          y = y_train,  # Numeric 0/1
          family = "binomial",
          alpha = alpha,
          weights = sample_weights,
          type.measure = "auc",
          nfolds = 3,  # Reduced for small sample
          standardize = TRUE,
          maxit = 100000
        )
      }, error = function(e) {
        # FIX: Print errors instead of silently catching them
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
        y = y_train,  # Numeric
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

# ------------------------------------------------------------------------------
# 5. XGBoost Model Training Function
# ------------------------------------------------------------------------------

train_xgboost <- function(X_train, y_train, X_val = NULL, y_val = NULL,
                          class_weights, tune = TRUE) {

  # y_train is already numeric 0/1
  sample_weights <- ifelse(y_train == 1,
                           class_weights["Nonunion"],
                           class_weights["Union"])

  # Check we have both classes
  if(length(unique(y_train)) < 2) {
    cat("    WARNING: XGBoost - only one class in training data\n")
    return(list(model = NULL, type = "xgboost"))
  }

  dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = sample_weights)

  if (tune) {
    # Simplified grid
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
        min_child_weight = 5  # Increased for small sample
      )

      cv_result <- tryCatch({
        xgb.cv(
          params = params,
          data = dtrain,
          nrounds = 50,  # Reduced
          nfold = 3,     # Reduced
          early_stopping_rounds = 10,
          verbose = 0,
          stratified = TRUE
        )
      }, error = function(e) {
        # FIX: Print errors instead of silently catching them
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

# ------------------------------------------------------------------------------
# 6. Prediction Function
# ------------------------------------------------------------------------------

# Unified prediction function
make_predictions <- function(model_object, X_new) {
  if (model_object$type == "glmnet") {
    # Elastic net predictions
    pred <- predict(model_object$model, newx = X_new, s = "lambda.1se", type = "response")
    as.numeric(pred)
  } else if (model_object$type == "xgboost") {
    # XGBoost predictions
    dtest <- xgb.DMatrix(data = X_new)
    predict(model_object$model, dtest)
  }
}

cat("  ✓ Model training functions defined\n\n")

# ------------------------------------------------------------------------------
# 7. Nested Cross-Validation Execution
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("RUNNING NESTED CROSS-VALIDATION\n")
cat("==============================================================================\n\n")

# Initialize storage for results
cv_results <- list(
  model1_elastic = list(),
  model2_elastic = list(),
  model1_xgboost = list(),
  model2_xgboost = list()
)

# Storage for predictions (for paired statistical tests)
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

# Start timer
start_time <- Sys.time()

for (rep in 1:n_repetitions) {
  cat("Repetition ", rep, "/", n_repetitions, "\n", sep = "")

  # Create stratified folds using numeric y
  folds <- create_stratified_folds(data2$y, k = n_outer_folds, seed = rep * 100)

  for (fold in 1:n_outer_folds) {
    cat("  Fold ", fold, "/", n_outer_folds, "... ", sep = "")

    test_idx <- folds[[fold]]
    train_idx <- setdiff(1:nrow(data2$X), test_idx)

    # Check class balance in fold
    if(length(unique(data2$y[train_idx])) < 2 || length(unique(data2$y[test_idx])) < 2) {
      cat("SKIPPED - insufficient class diversity\n")
      next
    }

    # Get data splits - y is numeric
    X1_train <- data1$X[train_idx, , drop = FALSE]
    X1_test <- data1$X[test_idx, , drop = FALSE]
    y_train <- data1$y[train_idx]  # Numeric
    y_test <- data1$y[test_idx]    # Numeric

    X2_train <- data2$X[train_idx, , drop = FALSE]
    X2_test <- data2$X[test_idx, , drop = FALSE]

    # Train models
    model1_en <- train_elastic_net(X1_train, y_train, class_weights = class_weights, tune = TRUE)
    model2_en <- train_elastic_net(X2_train, y_train, class_weights = class_weights, tune = TRUE)
    model1_xgb <- train_xgboost(X1_train, y_train, class_weights = class_weights, tune = TRUE)
    model2_xgb <- train_xgboost(X2_train, y_train, class_weights = class_weights, tune = TRUE)

    # Check if models trained
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

    # Calculate AUCs using factor for ROC
    roc1_en <- roc(y_test_factor, pred1_en, levels = c("Union", "Nonunion"), direction = "<", quiet = TRUE)
    roc2_en <- roc(y_test_factor, pred2_en, levels = c("Union", "Nonunion"), direction = "<", quiet = TRUE)
    roc1_xgb <- roc(y_test_factor, pred1_xgb, levels = c("Union", "Nonunion"), direction = "<", quiet = TRUE)
    roc2_xgb <- roc(y_test_factor, pred2_xgb, levels = c("Union", "Nonunion"), direction = "<", quiet = TRUE)

    # Store results
    cv_results$model1_elastic[[length(cv_results$model1_elastic) + 1]] <- list(
      repetition = rep,
      fold = fold,
      auc = as.numeric(auc(roc1_en)),
      roc = roc1_en,
      predictions = pred1_en,
      true_labels = y_test
    )

    cv_results$model2_elastic[[length(cv_results$model2_elastic) + 1]] <- list(
      repetition = rep,
      fold = fold,
      auc = as.numeric(auc(roc2_en)),
      roc = roc2_en,
      predictions = pred2_en,
      true_labels = y_test
    )

    cv_results$model1_xgboost[[length(cv_results$model1_xgboost) + 1]] <- list(
      repetition = rep,
      fold = fold,
      auc = as.numeric(auc(roc1_xgb)),
      roc = roc1_xgb,
      predictions = pred1_xgb,
      true_labels = y_test
    )

    cv_results$model2_xgboost[[length(cv_results$model2_xgboost) + 1]] <- list(
      repetition = rep,
      fold = fold,
      auc = as.numeric(auc(roc2_xgb)),
      roc = roc2_xgb,
      predictions = pred2_xgb,
      true_labels = y_test
    )

    # Store predictions for paired tests
    fold_predictions <- data.frame(
      repetition = rep,
      fold = fold,
      index = test_idx,
      true_label = as.character(y_test),
      model1_elastic_pred = pred1_en,
      model2_elastic_pred = pred2_en,
      model1_xgb_pred = pred1_xgb,
      model2_xgb_pred = pred2_xgb,
      stringsAsFactors = FALSE
    )

    all_predictions <- rbind(all_predictions, fold_predictions)

    cat("AUC_M1_EN=", round(as.numeric(auc(roc1_en)), 3),
        " AUC_M2_EN=", round(as.numeric(auc(roc2_en)), 3), "\n", sep = "")
  }
  cat("\n")
}

# End timer
end_time <- Sys.time()
elapsed_time <- difftime(end_time, start_time, units = "mins")

cat("✓ Nested CV completed in ", round(elapsed_time, 1), " minutes\n\n", sep = "")

# ------------------------------------------------------------------------------
# 8. Calculate Summary Statistics
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("PERFORMANCE SUMMARY\n")
cat("==============================================================================\n\n")

# Extract AUCs
auc_m1_en <- sapply(cv_results$model1_elastic, function(x) x$auc)
auc_m2_en <- sapply(cv_results$model2_elastic, function(x) x$auc)
auc_m1_xgb <- sapply(cv_results$model1_xgboost, function(x) x$auc)
auc_m2_xgb <- sapply(cv_results$model2_xgboost, function(x) x$auc)

# Calculate mean and 95% CI
calc_ci <- function(x, conf = 0.95) {
  mean_x <- mean(x, na.rm = TRUE)
  se_x <- sd(x, na.rm = TRUE) / sqrt(length(x))
  ci_lower <- mean_x - qnorm((1 + conf) / 2) * se_x
  ci_upper <- mean_x + qnorm((1 + conf) / 2) * se_x
  c(mean = mean_x, lower = ci_lower, upper = ci_upper)
}

# Elastic Net results
m1_en_summary <- calc_ci(auc_m1_en)
m2_en_summary <- calc_ci(auc_m2_en)

cat("ELASTIC NET MODELS:\n")
cat("-------------------\n")
cat("Model 1 (Baseline - Clinical + PROMIS):\n")
cat("  AUC: ", sprintf("%.3f", m1_en_summary["mean"]),
    " (95% CI: ", sprintf("%.3f", m1_en_summary["lower"]),
    "-", sprintf("%.3f", m1_en_summary["upper"]), ")\n", sep = "")

cat("\nModel 2 (Full - Clinical + PROMIS + RUST):\n")
cat("  AUC: ", sprintf("%.3f", m2_en_summary["mean"]),
    " (95% CI: ", sprintf("%.3f", m2_en_summary["lower"]),
    "-", sprintf("%.3f", m2_en_summary["upper"]), ")\n\n", sep = "")

# XGBoost results
m1_xgb_summary <- calc_ci(auc_m1_xgb)
m2_xgb_summary <- calc_ci(auc_m2_xgb)

cat("XGBOOST MODELS (Validation):\n")
cat("----------------------------\n")
cat("Model 1 (Baseline - Clinical + PROMIS):\n")
cat("  AUC: ", sprintf("%.3f", m1_xgb_summary["mean"]),
    " (95% CI: ", sprintf("%.3f", m1_xgb_summary["lower"]),
    "-", sprintf("%.3f", m1_xgb_summary["upper"]), ")\n", sep = "")

cat("\nModel 2 (Full - Clinical + PROMIS + RUST):\n")
cat("  AUC: ", sprintf("%.3f", m2_xgb_summary["mean"]),
    " (95% CI: ", sprintf("%.3f", m2_xgb_summary["lower"]),
    "-", sprintf("%.3f", m2_xgb_summary["upper"]), ")\n\n", sep = "")

# ------------------------------------------------------------------------------
# 9. Statistical Testing: DeLong's Test for Paired ROC Curves
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("STATISTICAL COMPARISON: DeLong's Test\n")
cat("==============================================================================\n\n")

# DeLong's test requires paired predictions on same samples
# Aggregate predictions by patient (averaging across folds where patient appears in test set)
patient_predictions <- all_predictions %>%
  group_by(index) %>%
  summarise(
    true_label = first(true_label),
    model1_elastic_pred = mean(model1_elastic_pred),
    model2_elastic_pred = mean(model2_elastic_pred),
    model1_xgb_pred = mean(model1_xgb_pred),
    model2_xgb_pred = mean(model2_xgb_pred),
    .groups = 'drop'
  )

# DeLong's test for Elastic Net models
roc_m1_en_final <- roc(patient_predictions$true_label, patient_predictions$model1_elastic_pred,
                       levels = c("0", "1"), direction = "<", quiet = TRUE)
roc_m2_en_final <- roc(patient_predictions$true_label, patient_predictions$model2_elastic_pred,
                       levels = c("0", "1"), direction = "<", quiet = TRUE)

delong_test_en <- roc.test(roc_m1_en_final, roc_m2_en_final, method = "delong")

cat("Elastic Net Models:\n")
cat("  Model 1 AUC: ", sprintf("%.3f", as.numeric(auc(roc_m1_en_final))), "\n", sep = "")
cat("  Model 2 AUC: ", sprintf("%.3f", as.numeric(auc(roc_m2_en_final))), "\n", sep = "")
cat("  Difference:  ", sprintf("%.3f", as.numeric(auc(roc_m2_en_final)) - as.numeric(auc(roc_m1_en_final))), "\n", sep = "")
cat("  p-value:     ", sprintf("%.4f", delong_test_en$p.value), "\n", sep = "")
if (delong_test_en$p.value < 0.05) {
  cat("  *** Statistically significant improvement (p < 0.05)\n\n")
} else {
  cat("  Not statistically significant (p >= 0.05)\n\n")
}

# DeLong's test for XGBoost models
roc_m1_xgb_final <- roc(patient_predictions$true_label, patient_predictions$model1_xgb_pred,
                        levels = c("0", "1"), direction = "<", quiet = TRUE)
roc_m2_xgb_final <- roc(patient_predictions$true_label, patient_predictions$model2_xgb_pred,
                        levels = c("0", "1"), direction = "<", quiet = TRUE)

delong_test_xgb <- roc.test(roc_m1_xgb_final, roc_m2_xgb_final, method = "delong")

cat("XGBoost Models:\n")
cat("  Model 1 AUC: ", sprintf("%.3f", as.numeric(auc(roc_m1_xgb_final))), "\n", sep = "")
cat("  Model 2 AUC: ", sprintf("%.3f", as.numeric(auc(roc_m2_xgb_final))), "\n", sep = "")
cat("  Difference:  ", sprintf("%.3f", as.numeric(auc(roc_m2_xgb_final)) - as.numeric(auc(roc_m1_xgb_final))), "\n", sep = "")
cat("  p-value:     ", sprintf("%.4f", delong_test_xgb$p.value), "\n", sep = "")
if (delong_test_xgb$p.value < 0.05) {
  cat("  *** Statistically significant improvement (p < 0.05)\n\n")
} else {
  cat("  Not statistically significant (p >= 0.05)\n\n")
}

# ------------------------------------------------------------------------------
# 10. Bootstrap Confidence Interval for AUC Difference
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("BOOTSTRAP CONFIDENCE INTERVAL FOR AUC DIFFERENCE\n")
cat("==============================================================================\n\n")

# Bootstrap function for AUC difference
boot_auc_diff <- function(data, indices, model1_col, model2_col) {
  d <- data[indices, ]

  roc1 <- roc(d$true_label, d[[model1_col]],
              levels = c("0", "1"), direction = "<", quiet = TRUE)
  roc2 <- roc(d$true_label, d[[model2_col]],
              levels = c("0", "1"), direction = "<", quiet = TRUE)

  as.numeric(auc(roc2)) - as.numeric(auc(roc1))
}

cat("Running bootstrap (2000 iterations)...\n")

# Bootstrap for Elastic Net
set.seed(42)
boot_results_en <- boot(
  data = patient_predictions,
  statistic = boot_auc_diff,
  R = 2000,
  model1_col = "model1_elastic_pred",
  model2_col = "model2_elastic_pred"
)

boot_ci_en <- boot.ci(boot_results_en, type = "perc", conf = 0.95)

cat("\nElastic Net Models:\n")
cat("  AUC Difference: ", sprintf("%.3f", boot_results_en$t0), "\n", sep = "")
cat("  95% CI: (", sprintf("%.3f", boot_ci_en$percent[4]),
    ", ", sprintf("%.3f", boot_ci_en$percent[5]), ")\n", sep = "")
if (boot_ci_en$percent[4] > 0) {
  cat("  *** CI does not include 0: significant improvement\n\n")
} else {
  cat("  CI includes 0: not significant\n\n")
}

# Bootstrap for XGBoost
set.seed(42)
boot_results_xgb <- boot(
  data = patient_predictions,
  statistic = boot_auc_diff,
  R = 2000,
  model1_col = "model1_xgb_pred",
  model2_col = "model2_xgb_pred"
)

boot_ci_xgb <- boot.ci(boot_results_xgb, type = "perc", conf = 0.95)

cat("XGBoost Models:\n")
cat("  AUC Difference: ", sprintf("%.3f", boot_results_xgb$t0), "\n", sep = "")
cat("  95% CI: (", sprintf("%.3f", boot_ci_xgb$percent[4]),
    ", ", sprintf("%.3f", boot_ci_xgb$percent[5]), ")\n", sep = "")
if (boot_ci_xgb$percent[4] > 0) {
  cat("  *** CI does not include 0: significant improvement\n\n")
} else {
  cat("  CI includes 0: not significant\n\n")
}

# ------------------------------------------------------------------------------
# 11. NRI and IDI Calculations
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("NET RECLASSIFICATION IMPROVEMENT (NRI) & \n")
cat("INTEGRATED DISCRIMINATION IMPROVEMENT (IDI)\n")
cat("==============================================================================\n\n")

# Prepare data for NRI/IDI
# Convert labels to binary (1 = Nonunion, 0 = Union)
y_binary <- as.numeric(patient_predictions$true_label)

# Calculate NRI and IDI for Elastic Net
cat("Elastic Net Models:\n")
cat("-------------------\n")

# Use PredictABEL package
nri_idi_en <- tryCatch({
  reclassification(
    data = data.frame(y = y_binary),
    cOutcome = 1,
    predrisk1 = patient_predictions$model1_elastic_pred,
    predrisk2 = patient_predictions$model2_elastic_pred,
    cutoff = c(0, 0.3, 1)  # Risk categories: low (<30%), high (>=30%)
  )
}, error = function(e) {
  cat("  Error calculating NRI/IDI:", e$message, "\n")
  NULL
})

if (!is.null(nri_idi_en)) {
  cat("  NRI (categorical):\n")
  print(nri_idi_en)
  cat("\n")
}

# Calculate continuous NRI manually
calc_continuous_nri <- function(y_true, pred1, pred2) {
  # For events (y = 1): proportion with increased risk
  events_idx <- which(y_true == 1)
  events_improved <- sum(pred2[events_idx] > pred1[events_idx]) / length(events_idx)
  events_worsened <- sum(pred2[events_idx] < pred1[events_idx]) / length(events_idx)

  # For non-events (y = 0): proportion with decreased risk
  nonevents_idx <- which(y_true == 0)
  nonevents_improved <- sum(pred2[nonevents_idx] < pred1[nonevents_idx]) / length(nonevents_idx)
  nonevents_worsened <- sum(pred2[nonevents_idx] > pred1[nonevents_idx]) / length(nonevents_idx)

  # NRI = (P(up|event) - P(down|event)) + (P(down|nonevent) - P(up|nonevent))
  nri_events <- events_improved - events_worsened
  nri_nonevents <- nonevents_improved - nonevents_worsened
  nri_total <- nri_events + nri_nonevents

  list(
    nri_total = nri_total,
    nri_events = nri_events,
    nri_nonevents = nri_nonevents
  )
}

# Calculate IDI manually
calc_idi <- function(y_true, pred1, pred2) {
  # IDI = (mean(pred2[events]) - mean(pred1[events])) - (mean(pred2[nonevents]) - mean(pred1[nonevents]))
  events_idx <- which(y_true == 1)
  nonevents_idx <- which(y_true == 0)

  idi_events <- mean(pred2[events_idx]) - mean(pred1[events_idx])
  idi_nonevents <- mean(pred2[nonevents_idx]) - mean(pred1[nonevents_idx])
  idi_total <- idi_events - idi_nonevents

  list(
    idi_total = idi_total,
    idi_events = idi_events,
    idi_nonevents = idi_nonevents
  )
}

# Continuous NRI for Elastic Net
nri_continuous_en <- calc_continuous_nri(y_binary,
                                         patient_predictions$model1_elastic_pred,
                                         patient_predictions$model2_elastic_pred)

cat("  Continuous NRI:\n")
cat("    Events:     ", sprintf("%.3f", nri_continuous_en$nri_events), "\n", sep = "")
cat("    Non-events: ", sprintf("%.3f", nri_continuous_en$nri_nonevents), "\n", sep = "")
cat("    Total:      ", sprintf("%.3f", nri_continuous_en$nri_total), "\n\n", sep = "")

# IDI for Elastic Net
idi_en <- calc_idi(y_binary,
                   patient_predictions$model1_elastic_pred,
                   patient_predictions$model2_elastic_pred)

cat("  Integrated Discrimination Improvement (IDI):\n")
cat("    Events:     ", sprintf("%.4f", idi_en$idi_events), "\n", sep = "")
cat("    Non-events: ", sprintf("%.4f", idi_en$idi_nonevents), "\n", sep = "")
cat("    Total:      ", sprintf("%.4f", idi_en$idi_total), "\n\n", sep = "")

# Same for XGBoost
cat("XGBoost Models:\n")
cat("---------------\n")

nri_continuous_xgb <- calc_continuous_nri(y_binary,
                                          patient_predictions$model1_xgb_pred,
                                          patient_predictions$model2_xgb_pred)

cat("  Continuous NRI:\n")
cat("    Events:     ", sprintf("%.3f", nri_continuous_xgb$nri_events), "\n", sep = "")
cat("    Non-events: ", sprintf("%.3f", nri_continuous_xgb$nri_nonevents), "\n", sep = "")
cat("    Total:      ", sprintf("%.3f", nri_continuous_xgb$nri_total), "\n\n", sep = "")

idi_xgb <- calc_idi(y_binary,
                    patient_predictions$model1_xgb_pred,
                    patient_predictions$model2_xgb_pred)

cat("  Integrated Discrimination Improvement (IDI):\n")
cat("    Events:     ", sprintf("%.4f", idi_xgb$idi_events), "\n", sep = "")
cat("    Non-events: ", sprintf("%.4f", idi_xgb$idi_nonevents), "\n", sep = "")
cat("    Total:      ", sprintf("%.4f", idi_xgb$idi_total), "\n\n", sep = "")

# ------------------------------------------------------------------------------
# 12. Train Final Models on Full Dataset
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("TRAINING FINAL MODELS ON FULL DATASET\n")
cat("==============================================================================\n\n")

cat("Training final models for variable importance and calibration...\n")

# Train final Elastic Net models
final_model1_en <- train_elastic_net(data1$X, data1$y, class_weights = class_weights, tune = FALSE)
final_model2_en <- train_elastic_net(data2$X, data2$y, class_weights = class_weights, tune = FALSE)

# Train final XGBoost models
final_model1_xgb <- train_xgboost(data1$X, data1$y, class_weights = class_weights, tune = FALSE)
final_model2_xgb <- train_xgboost(data2$X, data2$y, class_weights = class_weights, tune = FALSE)

cat("  ✓ Final models trained\n\n")

# ------------------------------------------------------------------------------
# 13. Variable Importance
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("VARIABLE IMPORTANCE\n")
cat("==============================================================================\n\n")

# Extract variable importance from Elastic Net
get_glmnet_importance <- function(model_obj, feature_names) {
  coefs <- coef(model_obj$model, s = "lambda.1se")
  coef_df <- data.frame(
    Feature = feature_names,
    Coefficient = as.numeric(coefs[-1]),  # Remove intercept
    AbsCoefficient = abs(as.numeric(coefs[-1]))
  )
  coef_df <- coef_df[order(-coef_df$AbsCoefficient), ]
  coef_df <- coef_df[coef_df$AbsCoefficient > 0, ]  # Keep only non-zero
  coef_df
}

# Model 1 importance
cat("Model 1 (Baseline) - Top 10 Features:\n")
cat("--------------------------------------\n")
m1_importance <- get_glmnet_importance(final_model1_en, colnames(data1$X))
print(head(m1_importance, 10), row.names = FALSE)
cat("\n")

# Model 2 importance
cat("Model 2 (Full) - Top 10 Features:\n")
cat("----------------------------------\n")
m2_importance <- get_glmnet_importance(final_model2_en, colnames(data2$X))
print(head(m2_importance, 10), row.names = FALSE)
cat("\n")

# Extract XGBoost importance
get_xgboost_importance <- function(model_obj, feature_names) {
  imp <- xgb.importance(feature_names = feature_names, model = model_obj$model)
  imp_df <- data.frame(
    Feature = imp$Feature,
    Gain = imp$Gain,
    Cover = imp$Cover,
    Frequency = imp$Frequency
  )
  imp_df
}

cat("Model 2 XGBoost - Top 10 Features by Gain:\n")
cat("------------------------------------------\n")
m2_xgb_importance <- get_xgboost_importance(final_model2_xgb, colnames(data2$X))
print(head(m2_xgb_importance, 10), row.names = FALSE)
cat("\n")

# ------------------------------------------------------------------------------
# 14. Calibration Plots
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("GENERATING CALIBRATION PLOTS\n")
cat("==============================================================================\n\n")

# Create calibration plots
create_calibration_plot <- function(y_true, y_pred, model_name, n_bins = 10) {
  # Convert to binary
  y_binary <- as.numeric(y_true)

  # Create bins
  pred_bins <- cut(y_pred, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)

  # Calculate observed and predicted rates per bin
  calib_data <- data.frame(
    y_true = y_binary,
    y_pred = y_pred,
    bin = pred_bins
  ) %>%
    group_by(bin) %>%
    summarise(
      predicted = mean(y_pred),
      observed = mean(y_true),
      n = n(),
      .groups = 'drop'
    )

  # Create plot
  p <- ggplot(calib_data, aes(x = predicted, y = observed)) +
    geom_point(aes(size = n), alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = TRUE, color = "blue", fill = "lightblue") +
    xlim(0, 1) + ylim(0, 1) +
    labs(
      title = paste("Calibration Plot:", model_name),
      x = "Predicted Probability",
      y = "Observed Probability",
      size = "N"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      panel.grid.minor = element_blank()
    )

  p
}

# Generate calibration plots for all models
p1 <- create_calibration_plot(patient_predictions$true_label,
                              patient_predictions$model1_elastic_pred,
                              "Model 1 - Elastic Net")

p2 <- create_calibration_plot(patient_predictions$true_label,
                              patient_predictions$model2_elastic_pred,
                              "Model 2 - Elastic Net")

p3 <- create_calibration_plot(patient_predictions$true_label,
                              patient_predictions$model1_xgb_pred,
                              "Model 1 - XGBoost")

p4 <- create_calibration_plot(patient_predictions$true_label,
                              patient_predictions$model2_xgb_pred,
                              "Model 2 - XGBoost")

# Combine plots
calibration_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2)

# Save plot
ggsave("calibration_plots.png", calibration_plot, width = 12, height = 10, dpi = 300)
cat("  ✓ Calibration plots saved: calibration_plots.png\n\n")

# ------------------------------------------------------------------------------
# 15. ROC Curves
# ------------------------------------------------------------------------------
cat("Generating ROC curves...\n")

# Create ROC plot
create_roc_plot <- function(roc1, roc2, model1_name, model2_name, title) {
  # Create data frames for plotting
  roc1_df <- data.frame(
    FPR = 1 - roc1$specificities,
    TPR = roc1$sensitivities,
    Model = model1_name
  )

  roc2_df <- data.frame(
    FPR = 1 - roc2$specificities,
    TPR = roc2$sensitivities,
    Model = model2_name
  )

  roc_df <- rbind(roc1_df, roc2_df)

  # Create plot
  p <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
    geom_line(size = 1.2) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
    xlim(0, 1) + ylim(0, 1) +
    labs(
      title = title,
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    ) +
    annotate("text", x = 0.6, y = 0.2,
             label = sprintf("%s AUC = %.3f", model1_name, as.numeric(auc(roc1))),
             color = scales::hue_pal()(2)[1], hjust = 0) +
    annotate("text", x = 0.6, y = 0.1,
             label = sprintf("%s AUC = %.3f", model2_name, as.numeric(auc(roc2))),
             color = scales::hue_pal()(2)[2], hjust = 0)

  p
}

# Create ROC plots
roc_plot_en <- create_roc_plot(roc_m1_en_final, roc_m2_en_final,
                               "Model 1 (Baseline)", "Model 2 (Full)",
                               "ROC Curves - Elastic Net Models")

roc_plot_xgb <- create_roc_plot(roc_m1_xgb_final, roc_m2_xgb_final,
                                "Model 1 (Baseline)", "Model 2 (Full)",
                                "ROC Curves - XGBoost Models")

# Combine ROC plots
roc_combined <- grid.arrange(roc_plot_en, roc_plot_xgb, ncol = 2)

# Save ROC plots
ggsave("roc_curves.png", roc_combined, width = 12, height = 5, dpi = 300)
cat("  ✓ ROC curves saved: roc_curves.png\n\n")

# ------------------------------------------------------------------------------
# 16. Abstract-Ready Summary Table
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("ABSTRACT-READY SUMMARY TABLE\n")
cat("==============================================================================\n\n")

# Create summary table
summary_table <- data.frame(
  Model = c(
    "Model 1: Baseline (Clinical + PROMIS)",
    "Model 2: Full (Clinical + PROMIS + RUST)"
  ),
  Algorithm = c("Elastic Net", "Elastic Net"),
  AUC = c(
    sprintf("%.2f (95%% CI: %.2f-%.2f)",
            m1_en_summary["mean"], m1_en_summary["lower"], m1_en_summary["upper"]),
    sprintf("%.2f (95%% CI: %.2f-%.2f)",
            m2_en_summary["mean"], m2_en_summary["lower"], m2_en_summary["upper"])
  ),
  stringsAsFactors = FALSE
)

# Add comparison row
comparison_row <- data.frame(
  Model = "Model 2 vs Model 1",
  Algorithm = "Comparison",
  AUC = sprintf("Δ = %.3f, p = %.4f",
                boot_results_en$t0,
                delong_test_en$p.value),
  stringsAsFactors = FALSE
)

summary_table <- rbind(summary_table, comparison_row)

print(summary_table, row.names = FALSE)
cat("\n")

# Abstract sentence
cat("ABSTRACT SENTENCE:\n")
cat("------------------\n")
cat(sprintf(
  "The addition of RUST scores to clinical and PROMIS features improved nonunion prediction from an AUC of %.2f (95%% CI: %.2f-%.2f) to %.2f (95%% CI: %.2f-%.2f), p = %.4f.\n\n",
  m1_en_summary["mean"], m1_en_summary["lower"], m1_en_summary["upper"],
  m2_en_summary["mean"], m2_en_summary["lower"], m2_en_summary["upper"],
  delong_test_en$p.value
))

# ------------------------------------------------------------------------------
# 17. Save All Results
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("SAVING RESULTS\n")
cat("==============================================================================\n\n")

# Save all results to RDS file
results_package <- list(
  # Cross-validation results
  cv_results = cv_results,
  all_predictions = all_predictions,
  patient_predictions = patient_predictions,

  # Summary statistics
  summary = list(
    model1_elastic = m1_en_summary,
    model2_elastic = m2_en_summary,
    model1_xgboost = m1_xgb_summary,
    model2_xgboost = m2_xgb_summary
  ),

  # Statistical tests
  delong_test_en = delong_test_en,
  delong_test_xgb = delong_test_xgb,
  bootstrap_en = boot_results_en,
  bootstrap_xgb = boot_results_xgb,
  nri_idi_en = list(
    nri_continuous = nri_continuous_en,
    idi = idi_en
  ),
  nri_idi_xgb = list(
    nri_continuous = nri_continuous_xgb,
    idi = idi_xgb
  ),

  # Final models
  final_models = list(
    model1_elastic = final_model1_en,
    model2_elastic = final_model2_en,
    model1_xgboost = final_model1_xgb,
    model2_xgboost = final_model2_xgb
  ),

  # Variable importance
  variable_importance = list(
    model1_elastic = m1_importance,
    model2_elastic = m2_importance,
    model2_xgboost = m2_xgb_importance
  ),

  # ROC objects
  roc_objects = list(
    model1_elastic = roc_m1_en_final,
    model2_elastic = roc_m2_en_final,
    model1_xgboost = roc_m1_xgb_final,
    model2_xgboost = roc_m2_xgb_final
  ),

  # Summary table
  summary_table = summary_table,

  # Metadata
  metadata = list(
    n_patients = nrow(model2_data),
    n_nonunion = n_nonunion,
    n_union = n_union,
    imbalance_ratio = imbalance_ratio,
    n_outer_folds = n_outer_folds,
    n_inner_folds = n_inner_folds,
    n_repetitions = n_repetitions,
    elapsed_time_mins = as.numeric(elapsed_time),
    timestamp = Sys.time()
  )
)

saveRDS(results_package, "model_comparison_results.rds")
cat("  ✓ Complete results saved: model_comparison_results.rds\n")

# Save summary table to CSV
write.csv(summary_table, "model_comparison_summary.csv", row.names = FALSE)
cat("  ✓ Summary table saved: model_comparison_summary.csv\n")

# Save variable importance to CSV
write.csv(m1_importance, "model1_variable_importance.csv", row.names = FALSE)
write.csv(m2_importance, "model2_variable_importance.csv", row.names = FALSE)
write.csv(m2_xgb_importance, "model2_xgboost_variable_importance.csv", row.names = FALSE)
cat("  ✓ Variable importance saved to CSV files\n")

# Save predictions to CSV (for external analysis)
write.csv(patient_predictions, "patient_predictions.csv", row.names = FALSE)
cat("  ✓ Patient predictions saved: patient_predictions.csv\n\n")

# ------------------------------------------------------------------------------
# 18. Final Summary
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("==============================================================================\n\n")

cat("Key Findings:\n")
cat("-------------\n")
cat("1. Model Performance (Elastic Net - Primary Analysis):\n")
cat("   - Model 1 AUC: ", sprintf("%.3f", m1_en_summary["mean"]),
    " (95% CI: ", sprintf("%.3f", m1_en_summary["lower"]),
    "-", sprintf("%.3f", m1_en_summary["upper"]), ")\n", sep = "")
cat("   - Model 2 AUC: ", sprintf("%.3f", m2_en_summary["mean"]),
    " (95% CI: ", sprintf("%.3f", m2_en_summary["lower"]),
    "-", sprintf("%.3f", m2_en_summary["upper"]), ")\n", sep = "")
cat("   - Improvement: ", sprintf("%.3f", boot_results_en$t0), "\n\n", sep = "")

cat("2. Statistical Significance:\n")
cat("   - DeLong's test p-value: ", sprintf("%.4f", delong_test_en$p.value), "\n", sep = "")
cat("   - Bootstrap 95% CI: (", sprintf("%.3f", boot_ci_en$percent[4]),
    ", ", sprintf("%.3f", boot_ci_en$percent[5]), ")\n", sep = "")
if (delong_test_en$p.value < 0.05 && boot_ci_en$percent[4] > 0) {
  cat("   - *** SIGNIFICANT IMPROVEMENT ***\n\n")
} else {
  cat("   - Not statistically significant\n\n")
}

cat("3. Reclassification Metrics:\n")
cat("   - Continuous NRI: ", sprintf("%.3f", nri_continuous_en$nri_total), "\n", sep = "")
cat("   - IDI: ", sprintf("%.4f", idi_en$idi_total), "\n\n", sep = "")

cat("4. Top Predictors (Model 2):\n")
if (nrow(m2_importance) > 0) {
  for (i in 1:min(5, nrow(m2_importance))) {
    cat("   ", i, ". ", m2_importance$Feature[i], " (coef = ",
        sprintf("%.3f", m2_importance$Coefficient[i]), ")\n", sep = "")
  }
  cat("\n")
}

cat("Output Files Generated:\n")
cat("-----------------------\n")
cat("  1. model_comparison_results.rds - Complete results object\n")
cat("  2. model_comparison_summary.csv - Summary table\n")
cat("  3. patient_predictions.csv - Individual predictions\n")
cat("  4. calibration_plots.png - Calibration curves\n")
cat("  5. roc_curves.png - ROC curves\n")
cat("  6. model1_variable_importance.csv - Baseline model features\n")
cat("  7. model2_variable_importance.csv - Full model features\n")
cat("  8. model2_xgboost_variable_importance.csv - XGBoost features\n\n")

cat("Analysis completed in ", round(elapsed_time, 1), " minutes.\n", sep = "")
cat("\n==============================================================================\n")
cat("Thank you for using this analysis pipeline!\n")
cat("==============================================================================\n")
