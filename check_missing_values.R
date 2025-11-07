# Quick script to check for missing values in the data

cat("Checking for missing values...\n\n")

# Load data
model1_data <- readRDS("poc_model1_data.rds")
model2_data <- readRDS("poc_model_data.rds")

# Check Model 1
cat("MODEL 1 DATA:\n")
cat("-------------\n")
cat("Dimensions: ", nrow(model1_data), " rows x ", ncol(model1_data), " columns\n")

# Count missing values
X1 <- as.matrix(model1_data[, -1])
y1 <- model1_data$Nonunion_Label

cat("\nMissing values in features:\n")
cat("  Total NAs in X: ", sum(is.na(X1)), "\n")
cat("  Rows with any NA: ", sum(apply(X1, 1, function(x) any(is.na(x)))), "\n")
cat("  Columns with any NA: ", sum(apply(X1, 2, function(x) any(is.na(x)))), "\n")

if (sum(is.na(X1)) > 0) {
  cat("\nColumns with missing values:\n")
  na_counts <- colSums(is.na(X1))
  na_cols <- na_counts[na_counts > 0]
  for (i in 1:min(10, length(na_cols))) {
    cat("  - ", names(na_cols)[i], ": ", na_cols[i], " NAs (",
        round(100 * na_cols[i] / nrow(X1), 1), "%)\n", sep="")
  }
  if (length(na_cols) > 10) {
    cat("  ... and ", length(na_cols) - 10, " more columns\n")
  }
}

cat("\nMissing values in outcome:\n")
cat("  NAs in y: ", sum(is.na(y1)), "\n\n")

# Check Model 2
cat("\nMODEL 2 DATA:\n")
cat("-------------\n")
cat("Dimensions: ", nrow(model2_data), " rows x ", ncol(model2_data), " columns\n")

X2 <- as.matrix(model2_data[, -1])
y2 <- model2_data$Nonunion_Label

cat("\nMissing values in features:\n")
cat("  Total NAs in X: ", sum(is.na(X2)), "\n")
cat("  Rows with any NA: ", sum(apply(X2, 1, function(x) any(is.na(x)))), "\n")
cat("  Columns with any NA: ", sum(apply(X2, 2, function(x) any(is.na(x)))), "\n")

if (sum(is.na(X2)) > 0) {
  cat("\nColumns with missing values:\n")
  na_counts <- colSums(is.na(X2))
  na_cols <- na_counts[na_counts > 0]
  for (i in 1:min(10, length(na_cols))) {
    cat("  - ", names(na_cols)[i], ": ", na_cols[i], " NAs (",
        round(100 * na_cols[i] / nrow(X2), 1), "%)\n", sep="")
  }
  if (length(na_cols) > 10) {
    cat("  ... and ", length(na_cols) - 10, " more columns\n")
  }
}

cat("\nMissing values in outcome:\n")
cat("  NAs in y: ", sum(is.na(y2)), "\n\n")

cat("==============================================================================\n")
cat("RECOMMENDATION:\n")
cat("==============================================================================\n")

total_na_X1 <- sum(is.na(X1))
total_na_X2 <- sum(is.na(X2))

if (total_na_X1 > 0 || total_na_X2 > 0) {
  cat("Missing values detected! You need to handle them before model training.\n\n")
  cat("Options:\n")
  cat("  1. Remove rows with missing values (listwise deletion)\n")
  cat("  2. Impute missing values (mean/median/mode imputation)\n")
  cat("  3. Use more advanced imputation (e.g., mice package)\n\n")

  rows_with_na_X1 <- sum(apply(X1, 1, function(x) any(is.na(x))))
  rows_with_na_X2 <- sum(apply(X2, 1, function(x) any(is.na(x))))

  cat("If using listwise deletion:\n")
  cat("  Model 1: Would lose ", rows_with_na_X1, " rows (",
      round(100 * rows_with_na_X1 / nrow(X1), 1), "%)\n", sep="")
  cat("  Model 2: Would lose ", rows_with_na_X2, " rows (",
      round(100 * rows_with_na_X2 / nrow(X2), 1), "%)\n\n", sep="")
} else {
  cat("No missing values found. Data is ready for model training.\n\n")
}
