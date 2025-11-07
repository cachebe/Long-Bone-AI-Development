# ==============================================================================
# Tibial Fracture Nonunion Prediction Model Comparison (v3 - MINIMAL FEATURES)
# ==============================================================================
# Purpose: Compare two prediction models using nested cross-validation:
#   - Model 1: Clinical + PROMIS (5 predictors)
#   - Model 2: Clinical + PROMIS + RUST (6 predictors)
#
# FIXES in v3:
#   - Minimal feature set (6 predictors max) to match sample size
#   - Reduced CV repetitions (5 instead of 10)
#   - Elastic net only (XGBoost removed - insufficient sample size)
#   - Focus on 1-3mo timepoint for early warning capability
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Load Required Libraries
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("TIBIAL NONUNION PREDICTION MODEL COMPARISON (v3 - MINIMAL)\n")
cat("==============================================================================\n\n")

cat("Loading required libraries...\n")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(caret)
  library(glmnet)
  library(pROC)
  library(ggplot2)
  library(boot)
})

options(scipen = 999)
set.seed(42)

cat("  ✓ All libraries loaded successfully\n\n")

# ------------------------------------------------------------------------------
# 2. Load Data and Create Minimal Feature Sets
# ------------------------------------------------------------------------------
cat("Loading and preparing minimal feature datasets...\n\n")

model2_data_full <- readRDS("poc_model_data.rds")

cat("Original dataset:\n")
cat("  - Total patients: ", nrow(model2_data_full), "\n", sep = "")
cat("  - Total features: ", ncol(model2_data_full) - 1, "\n", sep = "")
cat("  - Nonunion cases: ", sum(model2_data_full$Nonunion_Label == 1), "\n", sep = "")
cat("  - Events per feature ratio: ", 
    round(sum(model2_data_full$Nonunion_Label == 1) / (ncol(model2_data_full) - 1), 2), "\n\n", sep = "")

# Create minimal feature sets
cat("Creating minimal feature sets...\n")

# Model 1: Clinical + PROMIS only (5 predictors)
model1_data <- model2_data_full %>%
  select(
    Nonunion_Label,
    Age,
    BMI,
    ISS,
    IsOpen,
    PROMIS_PI_1_3mo
  )

# Model 2: Clinical + PROMIS + RUST (6 predictors)
model2_data <- model2_data_full %>%
  select(
    Nonunion_Label,
    Age,
    BMI,
    ISS,
    IsOpen,
    PROMIS_PI_1_3mo,
    RUST_Score_1to3
  )

cat("  ✓ Model 1 (Baseline): ", ncol(model1_data) - 1, " predictors\n", sep = "")
cat("    - Clinical: Age, BMI, ISS, IsOpen\n")
cat("    - PROMIS: PI at 1-3 months\n\n")

cat("  ✓ Model 2 (Full PoC): ", ncol(model2_data) - 1, " predictors\n", sep = "")
cat("    - Clinical: Age, BMI, ISS, IsOpen\n")
cat("    - PROMIS: PI at 1-3 months\n")
cat("    - RUST: Score at 1-3 months\n\n")

# Check for missing values
cat("Missing value check:\n")
cat("  - Model 1 missing: ", sum(is.na(model1_data)), "\n", sep = "")
cat("  - Model 2 missing: ", sum(is.na(model2_data)), "\n\n", sep = "")

# Class distribution
n_nonunion <- sum(model2_data$Nonunion_Label == 1)
n_union <- sum(model2_data$Nonunion_Label == 0)
imbalance_ratio <- n_union / n_nonunion

cat("Class distribution:\n")
cat("  - Nonunion (positive): ", n_nonunion, " (", 
    round(100 * n_nonunion / nrow(model2_data), 1), "%)\n", sep = "")
cat("  - Union (negative):    ", n_union, " (", 
    round(100 * n_union / nrow(model2_data), 1), "%)\n", sep = "")
cat("  - Imbalance ratio: ", round(imbalance_ratio, 2), ":1\n\n", sep = "")

cat("Statistical power assessment:\n")
cat("  - Events per feature (Model 1): ", round(n_nonunion / (ncol(model1_data) - 1), 1), "\n", sep = "")
cat("  - Events per feature (Model 2): ", round(n_nonunion / (ncol(model2_data) - 1), 1), "\n", sep = "")
cat("  - Recommended minimum: 10 events/feature\n")
cat("  - Status: ", 
    ifelse(n_nonunion / (ncol(model2_data) - 1) >= 10, "✓ ADEQUATE", "⚠ UNDERPOWERED"), 
    "\n\n", sep = "")

# Prepare data matrices
data1 <- list(
  X = as.matrix(model1_data[, -1]),
  y = as.numeric(model1_data$Nonunion_Label)
)

data2 <- list(
  X = as.matrix(model2_data[, -1]),
  y = as.numeric(model2_data$Nonunion_Label)
)

# Class weights for imbalanced data
class_weights <- c(
  "Union" = nrow(model2_data) / (2 * n_union),
  "Nonunion" = nrow(model2_data) / (2 * n_nonunion)
)

cat("Class weights for training:\n")
cat("  - Union weight:    ", round(class_weights["Union"], 3), "\n", sep = "")
cat("  - Nonunion weight: ", round(class_weights["Nonunion"], 3), "\n\n", sep = "")

# ------------------------------------------------------------------------------
# 3. Model Training Function (Elastic Net Only)
# ------------------------------------------------------------------------------
cat("Setting up elastic net training function...\n")

train_elastic_net <- function(X_train, y_train, class_weights) {
  
  # Verify no missing values
  if (any(is.na(X_train)) || any(is.na(y_train))) {
    cat("    ERROR: Missing values in training data!\n")
    return(NULL)
  }
  
  # Sample weights based on class
  sample_weights <- ifelse(y_train == 1,
                           class_weights["Nonunion"],
                           class_weights["Union"])
  
  # Grid search over alpha values
  alpha_grid <- seq(0, 1, by = 0.25)
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
      cat("    WARNING in cv.glmnet (alpha=", alpha, "): ", e$message, "\n", sep="")
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
    cv_auc = best_auc
  )
}

make_predictions <- function(model_object, X_new) {
  if (is.null(model_object$model)) {
    return(NULL)
  }
  pred <- predict(model_object$model, newx = X_new, s = "lambda.1se", type = "response")
  as.numeric(pred)
}

cat("  ✓ Training function defined\n\n")

# ------------------------------------------------------------------------------
# 4. Nested Cross-Validation Execution
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("RUNNING NESTED CROSS-VALIDATION\n")
cat("==============================================================================\n\n")

n_outer_folds <- 5
n_repetitions <- 5  # Reduced from 10

cat("CV Configuration:\n")
cat("  - Outer folds: ", n_outer_folds, "\n", sep = "")
cat("  - Repetitions: ", n_repetitions, "\n", sep = "")
cat("  - Total iterations: ", n_outer_folds * n_repetitions, "\n", sep = "")
cat("  - Expected events per test fold: ~", 
    round(n_nonunion / n_outer_folds, 1), " nonunions\n\n", sep = "")

# Storage for results
cv_results <- list(
  model1 = list(),
  model2 = list()
)

all_predictions <- data.frame(
  repetition = integer(),
  fold = integer(),
  index = integer(),
  true_label = integer(),
  model1_pred = numeric(),
  model2_pred = numeric(),
  stringsAsFactors = FALSE
)

start_time <- Sys.time()

for (rep in 1:n_repetitions) {
  cat("Repetition ", rep, "/", n_repetitions, "\n", sep = "")
  
  # Create stratified folds
  set.seed(rep * 100)
  folds <- createFolds(data2$y, k = n_outer_folds, list = TRUE, returnTrain = FALSE)
  
  for (fold in 1:n_outer_folds) {
    cat("  Fold ", fold, "/", n_outer_folds, "... ", sep = "")
    
    test_idx <- folds[[fold]]
    train_idx <- setdiff(1:nrow(data2$X), test_idx)
    
    # Check class diversity
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
    model1 <- train_elastic_net(X1_train, y_train, class_weights = class_weights)
    model2 <- train_elastic_net(X2_train, y_train, class_weights = class_weights)
    
    if (is.null(model1) || is.null(model2) || 
        is.null(model1$model) || is.null(model2$model)) {
      cat("FAILED - model training returned NULL\n")
      next
    }
    
    # Make predictions
    pred1 <- make_predictions(model1, X1_test)
    pred2 <- make_predictions(model2, X2_test)
    
    if (is.null(pred1) || is.null(pred2)) {
      cat("FAILED - prediction returned NULL\n")
      next
    }
    
    # Calculate AUCs
    y_test_factor <- factor(y_test, levels = c(0, 1), labels = c("Union", "Nonunion"))
    
    roc1 <- roc(y_test_factor, pred1, levels = c("Union", "Nonunion"), 
                direction = "<", quiet = TRUE)
    roc2 <- roc(y_test_factor, pred2, levels = c("Union", "Nonunion"), 
                direction = "<", quiet = TRUE)
    
    auc1 <- as.numeric(auc(roc1))
    auc2 <- as.numeric(auc(roc2))
    
    # Store results
    cv_results$model1[[length(cv_results$model1) + 1]] <- list(
      repetition = rep,
      fold = fold,
      auc = auc1,
      alpha = model1$alpha,
      cv_auc = model1$cv_auc,
      roc = roc1,
      predictions = pred1,
      true_labels = y_test,
      n_test = length(y_test),
      n_nonunion_test = sum(y_test == 1)
    )
    
    cv_results$model2[[length(cv_results$model2) + 1]] <- list(
      repetition = rep,
      fold = fold,
      auc = auc2,
      alpha = model2$alpha,
      cv_auc = model2$cv_auc,
      roc = roc2,
      predictions = pred2,
      true_labels = y_test,
      n_test = length(y_test),
      n_nonunion_test = sum(y_test == 1)
    )
    
    # Store predictions
    fold_predictions <- data.frame(
      repetition = rep,
      fold = fold,
      index = test_idx,
      true_label = y_test,
      model1_pred = pred1,
      model2_pred = pred2,
      stringsAsFactors = FALSE
    )
    all_predictions <- rbind(all_predictions, fold_predictions)
    
    cat("M1_AUC=", sprintf("%.3f", auc1), 
        " M2_AUC=", sprintf("%.3f", auc2),
        " (α1=", sprintf("%.2f", model1$alpha),
        " α2=", sprintf("%.2f", model2$alpha), ")\n", sep = "")
  }
  cat("\n")
}

end_time <- Sys.time()
elapsed_time <- difftime(end_time, start_time, units = "mins")

cat("✓ Nested CV completed in ", round(elapsed_time, 1), " minutes\n\n", sep = "")

# ------------------------------------------------------------------------------
# 5. Results Summary
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("CROSS-VALIDATION RESULTS SUMMARY\n")
cat("==============================================================================\n\n")

n_successful <- length(cv_results$model1)
cat("Successfully completed ", n_successful, " folds out of ", 
    n_repetitions * n_outer_folds, " total\n\n", sep = "")

if (n_successful == 0) {
  cat("ERROR: No folds completed successfully. Cannot compute results.\n")
  quit(save = "no", status = 1)
}

# Extract AUCs
aucs_m1 <- sapply(cv_results$model1, function(x) x$auc)
aucs_m2 <- sapply(cv_results$model2, function(x) x$auc)

# Summary statistics
cat("Model 1 (Clinical + PROMIS):\n")
cat("  - Mean AUC: ", sprintf("%.3f", mean(aucs_m1)), " ± ", 
    sprintf("%.3f", sd(aucs_m1)), "\n", sep = "")
cat("  - Median AUC: ", sprintf("%.3f", median(aucs_m1)), "\n", sep = "")
cat("  - Range: [", sprintf("%.3f", min(aucs_m1)), ", ", 
    sprintf("%.3f", max(aucs_m1)), "]\n\n", sep = "")

cat("Model 2 (Clinical + PROMIS + RUST):\n")
cat("  - Mean AUC: ", sprintf("%.3f", mean(aucs_m2)), " ± ", 
    sprintf("%.3f", sd(aucs_m2)), "\n", sep = "")
cat("  - Median AUC: ", sprintf("%.3f", median(aucs_m2)), "\n", sep = "")
cat("  - Range: [", sprintf("%.3f", min(aucs_m2)), ", ", 
    sprintf("%.3f", max(aucs_m2)), "]\n\n", sep = "")

# Improvement
auc_diff <- aucs_m2 - aucs_m1
cat("Model 2 vs Model 1:\n")
cat("  - Mean AUC improvement: ", sprintf("%.3f", mean(auc_diff)), " ± ", 
    sprintf("%.3f", sd(auc_diff)), "\n", sep = "")
cat("  - Folds where Model 2 improved: ", sum(auc_diff > 0), "/", n_successful, 
    " (", round(100 * sum(auc_diff > 0) / n_successful, 1), "%)\n\n", sep = "")

# Paired t-test
if (n_successful >= 3) {
  t_test <- t.test(aucs_m2, aucs_m1, paired = TRUE)
  cat("Paired t-test:\n")
  cat("  - t-statistic: ", sprintf("%.3f", t_test$statistic), "\n", sep = "")
  cat("  - p-value: ", sprintf("%.4f", t_test$p.value), "\n", sep = "")
  cat("  - 95% CI for difference: [", sprintf("%.3f", t_test$conf.int[1]), 
      ", ", sprintf("%.3f", t_test$conf.int[2]), "]\n", sep = "")
  cat("  - Interpretation: ", 
      ifelse(t_test$p.value < 0.05, 
             "✓ Significant improvement", 
             "No significant difference"), "\n\n", sep = "")
}

# Save results
saveRDS(list(
  cv_results = cv_results,
  all_predictions = all_predictions,
  summary = list(
    model1_aucs = aucs_m1,
    model2_aucs = aucs_m2,
    auc_differences = auc_diff,
    n_folds = n_successful
  )
), "cv_results_minimal.rds")

cat("✓ Results saved to: cv_results_minimal.rds\n\n")

# ------------------------------------------------------------------------------
# 6. Generate Summary Plot
# ------------------------------------------------------------------------------
cat("Generating results visualization...\n")

# Prepare data for plotting
plot_data <- data.frame(
  Model = rep(c("Model 1\n(Clinical + PROMIS)", 
                "Model 2\n(Clinical + PROMIS + RUST)"), 
              each = n_successful),
  AUC = c(aucs_m1, aucs_m2),
  Fold = rep(1:n_successful, 2)
)

# Box plot with individual points
p <- ggplot(plot_data, aes(x = Model, y = AUC, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", alpha = 0.5) +
  scale_fill_manual(values = c("#3498db", "#e74c3c")) +
  scale_y_continuous(limits = c(0.4, 1.0), breaks = seq(0.4, 1.0, 0.1)) +
  labs(
    title = "Tibial Nonunion Prediction Model Comparison",
    subtitle = paste0("Nested Cross-Validation (", n_successful, " folds)"),
    x = "",
    y = "AUC"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 11, face = "bold")
  )

ggsave("model_comparison_boxplot.png", p, width = 8, height = 6, dpi = 300)
cat("  ✓ Plot saved to: model_comparison_boxplot.png\n\n")

# ------------------------------------------------------------------------------
# 7. Feature Importance Analysis
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("FEATURE IMPORTANCE ANALYSIS\n")
cat("==============================================================================\n\n")

# Train final models on full dataset for coefficient extraction
cat("Training final models on full dataset...\n")

final_model1 <- train_elastic_net(data1$X, data1$y, class_weights)
final_model2 <- train_elastic_net(data2$X, data2$y, class_weights)

if (!is.null(final_model1$model) && !is.null(final_model2$model)) {
  
  # Extract coefficients
  coef1 <- as.matrix(coef(final_model1$model, s = "lambda.1se"))
  coef2 <- as.matrix(coef(final_model2$model, s = "lambda.1se"))
  
  cat("\nModel 1 Coefficients (Clinical + PROMIS):\n")
  print(round(coef1, 4))
  
  cat("\nModel 2 Coefficients (Clinical + PROMIS + RUST):\n")
  print(round(coef2, 4))
  
  cat("\n")
}

# ------------------------------------------------------------------------------
# 8. Final Summary
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("==============================================================================\n\n")

cat("Key Findings:\n")
cat("  1. Model 1 AUC: ", sprintf("%.3f ± %.3f", mean(aucs_m1), sd(aucs_m1)), "\n", sep = "")
cat("  2. Model 2 AUC: ", sprintf("%.3f ± %.3f", mean(aucs_m2), sd(aucs_m2)), "\n", sep = "")
cat("  3. Improvement: ", sprintf("%.3f", mean(auc_diff)), 
    " (", round(100 * mean(auc_diff) / mean(aucs_m1), 1), "%)\n", sep = "")

if (n_successful >= 3 && exists("t_test")) {
  cat("  4. Statistical significance: p = ", sprintf("%.4f", t_test$p.value), 
      " ", ifelse(t_test$p.value < 0.05, "(significant)", "(not significant)"), 
      "\n", sep = "")
}

cat("\nOutput files:\n")
cat("  - cv_results_minimal.rds (detailed results)\n")
cat("  - model_comparison_boxplot.png (visualization)\n\n")

cat("Interpretation for Abstract:\n")
if (mean(aucs_m2) > mean(aucs_m1) && (n_successful < 3 || t_test$p.value < 0.05)) {
  cat("  ✓ Adding RUST scores to clinical + PROMIS data significantly improved\n")
  cat("    nonunion prediction, supporting the hypothesis that radiographic\n")
  cat("    data adds predictive value beyond patient-reported outcomes.\n\n")
} else if (mean(aucs_m2) > mean(aucs_m1)) {
  cat("  ⚠ Adding RUST scores showed numerical improvement but did not reach\n")
  cat("    statistical significance. Larger sample size may be needed.\n\n")
} else {
  cat("  ⚠ No improvement detected. Consider:\n")
  cat("    - Verification of RUST score timing (must precede nonunion diagnosis)\n")
  cat("    - Alternative radiographic features\n")
  cat("    - Larger sample size for adequate statistical power\n\n")
}

cat("Next Steps:\n")
cat("  1. Review individual fold performance for outliers\n")
cat("  2. Perform DeLong test for formal statistical comparison\n")
cat("  3. Calculate NRI/IDI if improvement detected\n")
cat("  4. Draft abstract methods and results sections\n\n")

cat("Script completed successfully!\n")