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
library(janitor)
library(stringr)

cat("Libraries loaded successfully.\n\n")

pad_to_8_digits <- function(ids) {
  sprintf("%08s", str_pad(trimws(as.character(ids)), width = 8, side = "left", pad = "0"))
}

# ------------------------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------------------------
cat("Loading data files...\n")

# Step 2a: Load headers from the "Hidden" sheet (first row only)
cat("  - Loading headers from 'Hidden' sheet...\n")
header_data <- read_excel("2023/Tib Nonunion Combined.xlsx", sheet = "Analyze", skip = 234)
header_names <- colnames(header_data)
cat("    Found", length(header_names), "column headers\n")

# Step 2b: Load data from the "Analyze" sheet (NO headers in this sheet)
cat("  - Loading data from 'Analyze' sheet...\n")
data_main <- read_excel("2023/Tib Nonunion Combined.xlsx", sheet = "Analyze", col_names = FALSE)
data_main <- data_main %>% filter(row_number() <= n()-1)

cat("    Loaded", nrow(data_main), "rows,", ncol(data_main), "columns\n")

# Step 2c: Assign headers to the data
cat("  - Assigning headers to data...\n")
colnames(data_main) <- header_names
cat("    Headers assigned successfully\n")


cat("Data loading complete!\n")
cat("  - Main data:", nrow(data_main), "rows,", ncol(data_main), "columns\n")

# ------------------------------------------------------------------------------
# 3. Select and Rename Key Columns
# ------------------------------------------------------------------------------
cat("Selecting and renaming key columns...\n")

data_main_cleaned <- data_main %>%
  clean_names()

colnames(data_main_cleaned)

# Select and rename columns from main data using the NEW clean names
data_main_selected <- data_main_cleaned %>%
  mutate(pat_id = pad_to_8_digits(pat_id))%>%
  select(
    pat_id,
    Nonunion_Label = non_union_0_union_1_non_union,
    Age = age,
    BMI = bmi,
    pat_gender,
    cci_score, 
    # This is the corrected name from your console output:
    Smoking_Status = smoking_tobacco_use_0_no_2_yes, 
    GA_Open_Fracture = ga_open_frx_classification,
    AO_Classification = frx_classi,
    Tibia_Shaft_Classification = tib_shaft_class,
    ISS = iss,
    # These are the corrected mean PROMIS scores from your console output:
    PROMIS_PF_0_1mo = promis_pf_39,
    PROMIS_PI_0_1mo = pain_interference_40,
    PROMIS_Anxiety_0_1mo = anxiety_41, 
    PROMIS_PF_1_3mo = promis_pf_46,
    PROMIS_PI_1_3mo = pain_interference_47,
    PROMIS_Anxiety_1_3mo = anxiety_48, 
    PROMIS_PF_3_6mo = promis_pf_61,
    PROMIS_PI_3_6mo = pain_interference_62,
    PROMIS_Anxiety_3_6mo = anxiety_63,
    # These are the corrected RUST scores from your console output:
    RUST_Score_0to1 = rust_score_36,
    RUST_Score_1to3 = rust_score_43, 
    RUST_Score_3to6 = rust_score_58, 
    # follow up times
    fu_0_1 = f_u_time, 
    fu_1_3 = f_u_time_45, 
    fu_3_6 = f_u_time_60, 
  )

cat("  - Main data columns selected:", ncol(data_main_selected), "\n")

# ------------------------------------------------------------------------------
# 4. Merge Data
# ------------------------------------------------------------------------------
cat("Merging datasets...\n")

analysis_data <- data_main_selected 
colnames(analysis_data)

# ------------------------------------------------------------------------------
# 5. Aggressive Data Cleaning
# ------------------------------------------------------------------------------
cat("Performing aggressive data cleaning...\n")

# Define numeric columns (all except PAT_ID, AO_Classification, GA_Open_Fracture)
numeric_cols <- c("Nonunion_Label", "Age", "BMI", "cci_score", "Smoking_Status", "ISS",
                  "PROMIS_PF_0_1mo", "PROMIS_PI_0_1mo", "PROMIS_Anxiety_0_1mo", 
                  "PROMIS_PF_1_3mo", "PROMIS_PI_1_3mo", "PROMIS_Anxiety_1_3mo",
                  "PROMIS_PF_3_6mo", "PROMIS_PI_3_6mo", "PROMIS_Anxiety_3_6mo",
                  "RUST_Score_0to1", "RUST_Score_1to3", "RUST_Score_3to6", 
                  "fu_0_1", "fu_1_3", "fu_3_6")

# Convert to numeric and replace nonsensical values with NA
for (col in numeric_cols) {
  # Convert to numeric (this handles text errors like "#NUM!")
  analysis_data[[col]] <- suppressWarnings(as.numeric(analysis_data[[col]]))

  # Replace negative values (e.g., -9, -41, -999) with NA
  analysis_data[[col]][analysis_data[[col]] < 0] <- NA

  # Replace non-finite values with NA
  analysis_data[[col]][!is.finite(analysis_data[[col]])] <- NA
}

cat("  - Converted all numeric columns and replaced invalid values with NA\n")

# Handle Smoking_Status: Group prior (1) and current (2) as ever-smokers (1)
analysis_data <- analysis_data %>%
  mutate(Smoking_Status = case_when(
    Smoking_Status == 0 ~ 0,  # Never smoker
    Smoking_Status == 1 ~ 1,  # Prior smoker -> Ever smoker
    Smoking_Status == 2 ~ 1,  # Current smoker -> Ever smoker
    TRUE ~ NA_real_
  ))

cat("  - Recoded Smoking_Status: 0=never, 1=ever (prior or current)\n")

# Handle GA_Open_Fracture: Create binary IsOpen variable
analysis_data <- analysis_data %>%
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
analysis_data <- analysis_data %>%
  mutate(AO_Classification = as.character(AO_Classification))

# Create dummy variables for AO_Classification
data_with_dummies <- dummy_cols(
  analysis_data,
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
