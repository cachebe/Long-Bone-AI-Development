# ==============================================================================
# Nonunion Prediction Model - Data Preparation Script
# ==============================================================================
# This script prepares a clean, ML-ready dataset from two primary data sources:
# 1. Tib Nonunion Combined.xlsx - Main clinical and outcomes data
# 2. EDW481-DEMO_language update-20240131.xlsx - Demographics/comorbidity data
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Load Required Libraries
# ------------------------------------------------------------------------------
library(readxl)
library(dplyr)
library(tidyr)
library(fastDummies)

cat("Libraries loaded successfully.\n\n")

# ------------------------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------------------------
cat("Loading data files...\n")

# Step 2a: Load data from the "Analyze" sheet (NO headers - they're at the bottom)
cat("  - Loading data from 'Analyze' sheet...\n")
data_main_raw <- read_excel("2023/Tib Nonunion Combined.xlsx", sheet = "Analyze", col_names = FALSE)
cat("    Loaded", nrow(data_main_raw), "rows,", ncol(data_main_raw), "columns (includes header row at bottom)\n")

# Step 2b: Extract headers from the last row
cat("  - Extracting headers from bottom row...\n")
last_row_index <- nrow(data_main_raw)
header_names <- as.character(data_main_raw[last_row_index, ])
cat("    Found", length(header_names), "column headers\n")

# Step 2c: Remove the last row (header row) from the data
cat("  - Removing header row from data...\n")
data_main <- data_main_raw[-last_row_index, ]
cat("    Data now has", nrow(data_main), "rows (removed 1 header row)\n")

# Step 2d: Assign headers to the data
cat("  - Assigning headers to data...\n")
colnames(data_main) <- header_names
cat("    Headers assigned successfully\n")

# Step 2e: Load demographics/comorbidity data (first sheet)
cat("  - Loading demographics data...\n")
data_demo <- read_excel("2024/EDW481-DEMO_language update-20240131.xlsx", sheet = 1)
cat("    Loaded", nrow(data_demo), "rows,", ncol(data_demo), "columns\n\n")

cat("Data loading complete!\n")
cat("  - Main data:", nrow(data_main), "rows,", ncol(data_main), "columns\n")
cat("  - Demographics data:", nrow(data_demo), "rows,", ncol(data_demo), "columns\n\n")

# ------------------------------------------------------------------------------
# 3. Select and Rename Key Columns
# ------------------------------------------------------------------------------
cat("Selecting and renaming key columns...\n")

# Select and rename columns from main data
data_main_selected <- data_main %>%
  select(
    PAT_ID,
    Nonunion_Label = `Non-union? 0=union, 1=non-union`,
    Age,
    BMI,
    Smoking_Status = `SMOKING_TOBACCO_USE 0=never, 1=prior, 2= current`,
    GA_Open_Fracture = `GA Open Frx Classification`,
    AO_Classification = `Frx Classi`,
    ISS,
    PROMIS_PF_1_3mo = `Mean 1-3 PF`,
    PROMIS_PI_1_3mo = `Mean 1-3 PI`,
    PROMIS_PF_3_6mo = `Mean 3-6 PF`,
    PROMIS_PI_3_6mo = `Mean 3-6 PI`,
    RUST_Score_1to3 = `RUST Score 1to3`,
    RUST_Score_3to6 = `RUST Score 3to6`
  )

# Select columns from demographics data
data_demo_selected <- data_demo %>%
  select(PAT_ID, CCI_SCORE)

cat("  - Main data columns selected:", ncol(data_main_selected), "\n")
cat("  - Demographics columns selected:", ncol(data_demo_selected), "\n\n")

# ------------------------------------------------------------------------------
# 4. Merge Data
# ------------------------------------------------------------------------------
cat("Merging datasets...\n")

data_merged <- data_main_selected %>%
  left_join(data_demo_selected, by = "PAT_ID")

cat("  - Merged dataset:", nrow(data_merged), "rows,", ncol(data_merged), "columns\n\n")

# ------------------------------------------------------------------------------
# 5. Aggressive Data Cleaning
# ------------------------------------------------------------------------------
cat("Performing aggressive data cleaning...\n")

# Define numeric columns (all except PAT_ID, AO_Classification, GA_Open_Fracture)
numeric_cols <- c("Nonunion_Label", "Age", "BMI", "Smoking_Status", "ISS",
                  "PROMIS_PF_1_3mo", "PROMIS_PI_1_3mo",
                  "PROMIS_PF_3_6mo", "PROMIS_PI_3_6mo",
                  "RUST_Score_1to3", "RUST_Score_3to6", "CCI_SCORE")

# Convert to numeric and replace nonsensical values with NA
for (col in numeric_cols) {
  # Convert to numeric (this handles text errors like "#NUM!")
  data_merged[[col]] <- suppressWarnings(as.numeric(data_merged[[col]]))

  # Replace negative values (e.g., -9, -41, -999) with NA
  data_merged[[col]][data_merged[[col]] < 0] <- NA

  # Replace non-finite values with NA
  data_merged[[col]][!is.finite(data_merged[[col]])] <- NA
}

cat("  - Converted all numeric columns and replaced invalid values with NA\n")

# Handle Smoking_Status: Group prior (1) and current (2) as ever-smokers (1)
data_merged <- data_merged %>%
  mutate(Smoking_Status = case_when(
    Smoking_Status == 0 ~ 0,  # Never smoker
    Smoking_Status == 1 ~ 1,  # Prior smoker -> Ever smoker
    Smoking_Status == 2 ~ 1,  # Current smoker -> Ever smoker
    TRUE ~ NA_real_
  ))

cat("  - Recoded Smoking_Status: 0=never, 1=ever (prior or current)\n")

# Handle GA_Open_Fracture: Create binary IsOpen variable
data_merged <- data_merged %>%
  mutate(
    GA_Open_Fracture = as.character(GA_Open_Fracture),
    IsOpen = case_when(
      is.na(GA_Open_Fracture) ~ 0,
      GA_Open_Fracture %in% c("Closed", "0", "") ~ 0,
      TRUE ~ 1  # Any open grade (1, 2, 3A, 3B, etc.)
    )
  )

cat("  - Created binary IsOpen variable (0=closed, 1=any open fracture)\n")

# Handle AO_Classification: Create dummy variables
# First, clean and prepare the AO_Classification column
data_merged <- data_merged %>%
  mutate(AO_Classification = as.character(AO_Classification))

# Create dummy variables for AO_Classification
data_with_dummies <- dummy_cols(
  data_merged,
  select_columns = "AO_Classification",
  remove_first_dummy = FALSE,
  ignore_na = TRUE
)

cat("  - Created one-hot encoded variables for AO_Classification\n\n")

# ------------------------------------------------------------------------------
# 6. Impute Missing Data
# ------------------------------------------------------------------------------
cat("Checking missing data...\n")

# Count rows with any missing data
complete_rows <- sum(complete.cases(data_with_dummies))
total_rows <- nrow(data_with_dummies)
missing_rows <- total_rows - complete_rows

cat("  - Complete rows:", complete_rows, "/", total_rows, "\n")
cat("  - Rows with missing data:", missing_rows, "\n")

# Define predictor columns for imputation (numeric only, excluding outcome)
impute_cols <- c("Age", "BMI", "ISS", "CCI_SCORE",
                 "PROMIS_PF_1_3mo", "PROMIS_PI_1_3mo",
                 "PROMIS_PF_3_6mo", "PROMIS_PI_3_6mo",
                 "RUST_Score_1to3", "RUST_Score_3to6")

# Perform median imputation if there are missing values
if (missing_rows > 0) {
  cat("\nPerforming median imputation for numeric predictor columns...\n")

  for (col in impute_cols) {
    if (col %in% names(data_with_dummies)) {
      na_count <- sum(is.na(data_with_dummies[[col]]))
      if (na_count > 0) {
        median_val <- median(data_with_dummies[[col]], na.rm = TRUE)
        data_with_dummies[[col]][is.na(data_with_dummies[[col]])] <- median_val
        cat("  -", col, ": imputed", na_count, "values with median =", round(median_val, 2), "\n")
      }
    }
  }

  # Also impute Smoking_Status if needed
  if (sum(is.na(data_with_dummies$Smoking_Status)) > 0) {
    # Use mode (most common value) for binary variable
    mode_smoking <- as.numeric(names(sort(table(data_with_dummies$Smoking_Status), decreasing = TRUE)[1]))
    na_count <- sum(is.na(data_with_dummies$Smoking_Status))
    data_with_dummies$Smoking_Status[is.na(data_with_dummies$Smoking_Status)] <- mode_smoking
    cat("  - Smoking_Status: imputed", na_count, "values with mode =", mode_smoking, "\n")
  }

  # Impute IsOpen if needed
  if (sum(is.na(data_with_dummies$IsOpen)) > 0) {
    mode_isopen <- as.numeric(names(sort(table(data_with_dummies$IsOpen), decreasing = TRUE)[1]))
    na_count <- sum(is.na(data_with_dummies$IsOpen))
    data_with_dummies$IsOpen[is.na(data_with_dummies$IsOpen)] <- mode_isopen
    cat("  - IsOpen: imputed", na_count, "values with mode =", mode_isopen, "\n")
  }

  cat("\nImputation complete!\n\n")
} else {
  cat("  - No imputation needed, all data complete!\n\n")
}

# ------------------------------------------------------------------------------
# 7. Finalize Datasets for Models
# ------------------------------------------------------------------------------
cat("Creating model-ready datasets...\n")

# Identify AO dummy columns
ao_dummy_cols <- grep("^AO_Classification_", names(data_with_dummies), value = TRUE)

# Define clinical predictors (excluding RUST scores)
clinical_predictors <- c("Age", "BMI", "Smoking_Status", "ISS", "CCI_SCORE", "IsOpen")

# Define PROMIS predictors
promis_predictors <- c("PROMIS_PF_1_3mo", "PROMIS_PI_1_3mo",
                       "PROMIS_PF_3_6mo", "PROMIS_PI_3_6mo")

# Define RUST predictors
rust_predictors <- c("RUST_Score_1to3", "RUST_Score_3to6")

# Model 1: Baseline PoC (Clinical + PROMIS + AO dummies)
model1_data <- data_with_dummies %>%
  select(
    Nonunion_Label,
    all_of(clinical_predictors),
    all_of(promis_predictors),
    all_of(ao_dummy_cols)
  )

cat("  - Model 1 (Baseline PoC): ", ncol(model1_data), "columns (",
    ncol(model1_data) - 1, "predictors + outcome)\n")
cat("    Predictors: Clinical + PROMIS + AO Classification\n")

# Model 2: Full PoC (Clinical + PROMIS + RUST + AO dummies)
model2_data <- data_with_dummies %>%
  select(
    Nonunion_Label,
    all_of(clinical_predictors),
    all_of(promis_predictors),
    all_of(rust_predictors),
    all_of(ao_dummy_cols)
  )

cat("  - Model 2 (Full PoC): ", ncol(model2_data), "columns (",
    ncol(model2_data) - 1, "predictors + outcome)\n")
cat("    Predictors: Clinical + PROMIS + RUST + AO Classification\n\n")

# Check for any remaining missing values
cat("Final data quality check:\n")
cat("  - Model 1 missing values:", sum(is.na(model1_data)), "\n")
cat("  - Model 2 missing values:", sum(is.na(model2_data)), "\n")
cat("  - Model 1 complete cases:", sum(complete.cases(model1_data)), "/", nrow(model1_data), "\n")
cat("  - Model 2 complete cases:", sum(complete.cases(model2_data)), "/", nrow(model2_data), "\n\n")

# ------------------------------------------------------------------------------
# 8. Save Output
# ------------------------------------------------------------------------------
cat("Saving final dataset...\n")

# Save the full PoC model data
saveRDS(model2_data, "poc_model_data.rds")
cat("  - Saved: poc_model_data.rds\n")

# Optional: Also save model1_data for comparison
saveRDS(model1_data, "poc_model1_data.rds")
cat("  - Saved: poc_model1_data.rds (bonus - baseline model)\n\n")

# ------------------------------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("DATA PREPARATION COMPLETE!\n")
cat("==============================================================================\n\n")

cat("Summary:\n")
cat("  - Total patients:", nrow(model2_data), "\n")
cat("  - Nonunion cases:", sum(model2_data$Nonunion_Label == 1, na.rm = TRUE), "\n")
cat("  - Union cases:", sum(model2_data$Nonunion_Label == 0, na.rm = TRUE), "\n")
cat("  - Nonunion rate:",
    round(100 * sum(model2_data$Nonunion_Label == 1, na.rm = TRUE) / nrow(model2_data), 1), "%\n\n")

cat("Output files created:\n")
cat("  - poc_model_data.rds (Full PoC model - Clinical + PROMIS + RUST)\n")
cat("  - poc_model1_data.rds (Baseline PoC model - Clinical + PROMIS)\n\n")

cat("Next steps:\n")
cat("  1. Load data: model_data <- readRDS('poc_model_data.rds')\n")
cat("  2. Split into train/test sets\n")
cat("  3. Train your ML models (e.g., logistic regression, random forest, XGBoost)\n")
cat("  4. Evaluate performance (AUC, sensitivity, specificity, etc.)\n\n")

cat("Good luck with your modeling!\n")
