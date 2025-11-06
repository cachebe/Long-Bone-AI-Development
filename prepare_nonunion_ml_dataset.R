# ==============================================================================
# Nonunion Prediction Model - Data Preparation Script (Enhanced Version)
# ==============================================================================
# This script prepares a clean, ML-ready dataset from three primary data sources:
# 1. Tib Nonunion Combined.xlsx - Main clinical and outcomes data
# 2. EDW481-DEMO_language update-20240131.xlsx - Demographics/comorbidity data
# 3. Tib_Nonunion_RUST.xlsx - RUST scores (longitudinal radiographic data)
#
# KEY INNOVATION: Sophisticated imputation that differentiates between:
# - Clinical variables (MCAR): Median/mode imputation
# - Longitudinal variables (MNAR): Fixed "bad" values + missingness flags
# ==============================================================================

# ------------------------------------------------------------------------------
# Block 1: Load Required Libraries
# ------------------------------------------------------------------------------
library(readxl)
library(dplyr)
library(tidyr)
library(fastDummies)
library(janitor)
library(stringr)

cat("Libraries loaded successfully.\n\n")

# Helper function to pad patient IDs to 8 digits
pad_to_8_digits <- function(ids) {
  sprintf("%08s", str_pad(trimws(as.character(ids)), width = 8, side = "left", pad = "0"))
}

# ------------------------------------------------------------------------------
# Block 2: Load Data from Multiple Sources
# ------------------------------------------------------------------------------
cat("Loading data files...\n")

# Step 2a: Load data from the "Analyze" sheet (headers at bottom)
cat("  - Loading main data from 'Analyze' sheet...\n")
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
cat("    Data now has", nrow(data_main), "rows\n")

# Step 2d: Assign headers to the data
cat("  - Assigning headers to data...\n")
colnames(data_main) <- header_names
cat("    Headers assigned successfully\n")

# Step 2e: Load RUST data (longitudinal radiographic scores)
cat("  - Loading RUST data from separate file...\n")
data_rust <- read_excel("2023/Tib_Nonunion_RUST.xlsx", sheet = 1)
cat("    Loaded", nrow(data_rust), "rows,", ncol(data_rust), "columns\n")
cat("    RUST columns:", paste(colnames(data_rust), collapse = ", "), "\n")

cat("\nData loading complete!\n")
cat("  - Main data:", nrow(data_main), "rows,", ncol(data_main), "columns\n")
cat("  - RUST data:", nrow(data_rust), "rows,", ncol(data_rust), "columns\n\n")

# ------------------------------------------------------------------------------
# Block 3: Select and Rename Key Columns
# ------------------------------------------------------------------------------
cat("Selecting and renaming key columns...\n")

# Clean column names
data_main_cleaned <- data_main %>%
  clean_names()

# Select and rename columns from main data
data_main_selected <- data_main_cleaned %>%
  mutate(pat_id = pad_to_8_digits(pat_id)) %>%
  select(
    pat_id,
    Nonunion_Label = non_union_0_union_1_non_union,
    Age = age,
    BMI = bmi,
    pat_gender,
    CCI_SCORE = cci_score,
    Smoking_Status = smoking_tobacco_use_0_no_2_yes,
    GA_Open_Fracture = ga_open_frx_classification,
    AO_Classification = frx_classi,
    Tibia_Shaft_Classification = tib_shaft_class,
    ISS = iss,
    # PROMIS scores at different time points
    PROMIS_PF_0_1mo = promis_pf_39,
    PROMIS_PI_0_1mo = pain_interference_40,
    PROMIS_Anxiety_0_1mo = anxiety_41,
    PROMIS_PF_1_3mo = promis_pf_46,
    PROMIS_PI_1_3mo = pain_interference_47,
    PROMIS_Anxiety_1_3mo = anxiety_48,
    PROMIS_PF_3_6mo = promis_pf_61,
    PROMIS_PI_3_6mo = pain_interference_62,
    PROMIS_Anxiety_3_6mo = anxiety_63,
    # Follow up times
    fu_0_1 = f_u_time,
    fu_1_3 = f_u_time_45,
    fu_3_6 = f_u_time_60
  )

cat("  - Main data columns selected:", ncol(data_main_selected), "\n")

# Clean RUST data column names and pad patient IDs
data_rust_cleaned <- data_rust %>%
  clean_names() %>%
  mutate(pat_id = pad_to_8_digits(pat_id))

cat("  - RUST data cleaned and patient IDs padded\n")

# ------------------------------------------------------------------------------
# Block 4: Merge RUST Data with Main Data
# ------------------------------------------------------------------------------
cat("\nMerging RUST data with main dataset...\n")

# Pivot RUST data from long to wide format
# Assuming RUST data has: PAT_ID, follow_up_window, RUST_Score
# follow_up_window values like: "1-3mo", "3-6mo", "0-1mo"
data_rust_wide <- data_rust_cleaned %>%
  select(pat_id, follow_up_window, rust_score) %>%
  mutate(
    # Create standardized window labels
    time_window = case_when(
      grepl("0.*1|baseline", follow_up_window, ignore.case = TRUE) ~ "0to1",
      grepl("1.*3", follow_up_window, ignore.case = TRUE) ~ "1to3",
      grepl("3.*6", follow_up_window, ignore.case = TRUE) ~ "3to6",
      TRUE ~ as.character(follow_up_window)
    )
  ) %>%
  select(pat_id, time_window, rust_score) %>%
  pivot_wider(
    names_from = time_window,
    values_from = rust_score,
    names_prefix = "RUST_Score_"
  )

cat("  - Pivoted RUST data to wide format\n")
cat("  - RUST columns created:", paste(grep("^RUST_Score_", names(data_rust_wide), value = TRUE), collapse = ", "), "\n")

# Merge with main data
analysis_data <- data_main_selected %>%
  left_join(data_rust_wide, by = "pat_id")

cat("  - Merged RUST data with main dataset\n")
cat("  - Final dataset:", nrow(analysis_data), "rows,", ncol(analysis_data), "columns\n\n")

# ------------------------------------------------------------------------------
# Block 5: Aggressive Data Cleaning
# ------------------------------------------------------------------------------
cat("Performing aggressive data cleaning...\n")

# Define numeric columns (all except PAT_ID, categorical variables)
numeric_cols <- c(
  "Nonunion_Label", "Age", "BMI", "CCI_SCORE", "Smoking_Status", "ISS",
  "PROMIS_PF_0_1mo", "PROMIS_PI_0_1mo", "PROMIS_Anxiety_0_1mo",
  "PROMIS_PF_1_3mo", "PROMIS_PI_1_3mo", "PROMIS_Anxiety_1_3mo",
  "PROMIS_PF_3_6mo", "PROMIS_PI_3_6mo", "PROMIS_Anxiety_3_6mo",
  "fu_0_1", "fu_1_3", "fu_3_6"
)

# Add RUST columns dynamically (in case they have different names)
rust_cols <- grep("^RUST_Score_", names(analysis_data), value = TRUE)
numeric_cols <- c(numeric_cols, rust_cols)

# Convert to numeric and replace nonsensical values with NA
for (col in numeric_cols) {
  if (col %in% names(analysis_data)) {
    # Convert to numeric (this handles text errors like "#NUM!")
    analysis_data[[col]] <- suppressWarnings(as.numeric(analysis_data[[col]]))

    # Replace negative values (e.g., -9, -41, -999) with NA
    analysis_data[[col]][analysis_data[[col]] < 0] <- NA

    # Replace non-finite values with NA
    analysis_data[[col]][!is.finite(analysis_data[[col]])] <- NA
  }
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
# Block 6: Sophisticated Imputation Strategy
# ------------------------------------------------------------------------------
cat("==============================================================================\n")
cat("SOPHISTICATED IMPUTATION FOR MNAR DATA\n")
cat("==============================================================================\n\n")

cat("Strategy:\n")
cat("  1. CLINICAL VARS (likely MCAR): Impute with median/mode\n")
cat("     - Age, BMI, ISS, CCI_SCORE, Smoking_Status, IsOpen\n")
cat("  2. LONGITUDINAL VARS (likely MNAR): Fixed 'bad' value + missingness flag\n")
cat("     - PROMIS scores, RUST scores\n")
cat("     - Rationale: Missing follow-up may predict worse outcomes\n\n")

# Count initial missing data
complete_rows <- sum(complete.cases(data_with_dummies))
total_rows <- nrow(data_with_dummies)
missing_rows <- total_rows - complete_rows

cat("Initial data quality:\n")
cat("  - Complete rows:", complete_rows, "/", total_rows, "\n")
cat("  - Rows with missing data:", missing_rows, "\n\n")

# ---- Step 6A: Impute Clinical Variables (MCAR) ----
cat("Step 6A: Imputing CLINICAL variables with median/mode...\n")

clinical_vars <- c("Age", "BMI", "ISS", "CCI_SCORE")

for (col in clinical_vars) {
  if (col %in% names(data_with_dummies)) {
    na_count <- sum(is.na(data_with_dummies[[col]]))
    if (na_count > 0) {
      median_val <- median(data_with_dummies[[col]], na.rm = TRUE)
      data_with_dummies[[col]][is.na(data_with_dummies[[col]])] <- median_val
      cat("  -", col, ": imputed", na_count, "values with median =", round(median_val, 2), "\n")
    }
  }
}

# Impute Smoking_Status with mode
if (sum(is.na(data_with_dummies$Smoking_Status)) > 0) {
  mode_smoking <- as.numeric(names(sort(table(data_with_dummies$Smoking_Status), decreasing = TRUE)[1]))
  na_count <- sum(is.na(data_with_dummies$Smoking_Status))
  data_with_dummies$Smoking_Status[is.na(data_with_dummies$Smoking_Status)] <- mode_smoking
  cat("  - Smoking_Status: imputed", na_count, "values with mode =", mode_smoking, "\n")
}

# Impute IsOpen with mode
if (sum(is.na(data_with_dummies$IsOpen)) > 0) {
  mode_isopen <- as.numeric(names(sort(table(data_with_dummies$IsOpen), decreasing = TRUE)[1]))
  na_count <- sum(is.na(data_with_dummies$IsOpen))
  data_with_dummies$IsOpen[is.na(data_with_dummies$IsOpen)] <- mode_isopen
  cat("  - IsOpen: imputed", na_count, "values with mode =", mode_isopen, "\n")
}

cat("\n")

# ---- Step 6B: Impute Longitudinal Variables (MNAR) ----
cat("Step 6B: Imputing LONGITUDINAL variables with fixed 'bad' values + flags...\n\n")

# Define longitudinal variables and their "bad" imputation values
longitudinal_vars <- list(
  # PROMIS Physical Function: Lower is worse, use 20
  list(var = "PROMIS_PF_1_3mo", bad_value = 20, description = "PROMIS PF 1-3mo (low=bad)"),
  list(var = "PROMIS_PF_3_6mo", bad_value = 20, description = "PROMIS PF 3-6mo (low=bad)"),

  # PROMIS Pain Interference: Higher is worse, use 70
  list(var = "PROMIS_PI_1_3mo", bad_value = 70, description = "PROMIS PI 1-3mo (high=bad)"),
  list(var = "PROMIS_PI_3_6mo", bad_value = 70, description = "PROMIS PI 3-6mo (high=bad)"),

  # PROMIS Anxiety: Higher is worse, use 70
  list(var = "PROMIS_Anxiety_1_3mo", bad_value = 70, description = "PROMIS Anxiety 1-3mo (high=bad)"),
  list(var = "PROMIS_Anxiety_3_6mo", bad_value = 70, description = "PROMIS Anxiety 3-6mo (high=bad)"),

  # RUST Score: Lower is worse (range 4-12), use 4
  list(var = "RUST_Score_1to3", bad_value = 4, description = "RUST 1-3mo (low=bad, 4=worst)"),
  list(var = "RUST_Score_3to6", bad_value = 4, description = "RUST 3-6mo (low=bad, 4=worst)")
)

# Add any additional RUST columns dynamically
other_rust_cols <- setdiff(rust_cols, c("RUST_Score_1to3", "RUST_Score_3to6"))
for (rust_col in other_rust_cols) {
  longitudinal_vars[[length(longitudinal_vars) + 1]] <- list(
    var = rust_col,
    bad_value = 4,
    description = paste(rust_col, "(low=bad, 4=worst)")
  )
}

# Process each longitudinal variable
for (var_info in longitudinal_vars) {
  var_name <- var_info$var
  bad_value <- var_info$bad_value
  description <- var_info$description

  if (var_name %in% names(data_with_dummies)) {
    # Create missingness flag column
    flag_col <- paste0(var_name, "_IsMissing")
    data_with_dummies[[flag_col]] <- ifelse(is.na(data_with_dummies[[var_name]]), 1, 0)

    # Count missing values
    na_count <- sum(is.na(data_with_dummies[[var_name]]))

    # Impute with fixed "bad" value
    data_with_dummies[[var_name]][is.na(data_with_dummies[[var_name]])] <- bad_value

    cat("  -", description, "\n")
    cat("    * Imputed", na_count, "NAs with fixed value =", bad_value, "\n")
    cat("    * Created flag:", flag_col, "(1=missing, 0=observed)\n\n")
  }
}

cat("Imputation complete!\n\n")

# ---- Step 6C: Summary Statistics ----
cat("Post-imputation data quality:\n")
complete_rows_after <- sum(complete.cases(data_with_dummies))
cat("  - Complete rows:", complete_rows_after, "/", total_rows, "\n")
cat("  - Remaining missing values:", sum(is.na(data_with_dummies)), "\n\n")

# ------------------------------------------------------------------------------
# Block 7: Finalize Datasets for Models
# ------------------------------------------------------------------------------
cat("Creating model-ready datasets...\n")

# Identify AO dummy columns
ao_dummy_cols <- grep("^AO_Classification_", names(data_with_dummies), value = TRUE)

# Identify missingness flag columns
missingness_flags <- grep("_IsMissing$", names(data_with_dummies), value = TRUE)

# Define clinical predictors
clinical_predictors <- c("Age", "BMI", "Smoking_Status", "ISS", "CCI_SCORE", "IsOpen")

# Define PROMIS predictors (imputed values)
promis_predictors <- c(
  "PROMIS_PF_1_3mo", "PROMIS_PI_1_3mo", "PROMIS_Anxiety_1_3mo",
  "PROMIS_PF_3_6mo", "PROMIS_PI_3_6mo", "PROMIS_Anxiety_3_6mo"
)

# Define PROMIS missingness flags
promis_flags <- grep("^PROMIS.*_IsMissing$", missingness_flags, value = TRUE)

# Define RUST predictors (imputed values)
rust_predictors <- grep("^RUST_Score_", names(data_with_dummies), value = TRUE)
rust_predictors <- rust_predictors[!grepl("_IsMissing$", rust_predictors)]

# Define RUST missingness flags
rust_flags <- grep("^RUST.*_IsMissing$", missingness_flags, value = TRUE)

# Model 1: Baseline PoC (Clinical + PROMIS + PROMIS flags + AO dummies)
model1_data <- data_with_dummies %>%
  select(
    Nonunion_Label,
    all_of(clinical_predictors),
    all_of(promis_predictors),
    all_of(promis_flags),
    all_of(ao_dummy_cols)
  )

cat("  - Model 1 (Baseline PoC): ", ncol(model1_data), "columns (",
    ncol(model1_data) - 1, "predictors + outcome)\n")
cat("    Predictors: Clinical + PROMIS (imputed) + PROMIS flags + AO Classification\n")
cat("    Missingness flags included:", length(promis_flags), "\n")

# Model 2: Full PoC (Clinical + PROMIS + PROMIS flags + RUST + RUST flags + AO dummies)
model2_data <- data_with_dummies %>%
  select(
    Nonunion_Label,
    all_of(clinical_predictors),
    all_of(promis_predictors),
    all_of(promis_flags),
    all_of(rust_predictors),
    all_of(rust_flags),
    all_of(ao_dummy_cols)
  )

cat("  - Model 2 (Full PoC): ", ncol(model2_data), "columns (",
    ncol(model2_data) - 1, "predictors + outcome)\n")
cat("    Predictors: Clinical + PROMIS (imputed) + PROMIS flags + RUST (imputed) + RUST flags + AO Classification\n")
cat("    Missingness flags included:", length(promis_flags) + length(rust_flags), "\n\n")

# Check for any remaining missing values
cat("Final data quality check:\n")
cat("  - Model 1 missing values:", sum(is.na(model1_data)), "\n")
cat("  - Model 2 missing values:", sum(is.na(model2_data)), "\n")
cat("  - Model 1 complete cases:", sum(complete.cases(model1_data)), "/", nrow(model1_data), "\n")
cat("  - Model 2 complete cases:", sum(complete.cases(model2_data)), "/", nrow(model2_data), "\n\n")

# ------------------------------------------------------------------------------
# Block 8: Save Output
# ------------------------------------------------------------------------------
cat("Saving final datasets...\n")

# Save the full PoC model data
saveRDS(model2_data, "poc_model_data.rds")
cat("  - Saved: poc_model_data.rds\n")

# Save baseline model data
saveRDS(model1_data, "poc_model1_data.rds")
cat("  - Saved: poc_model1_data.rds (baseline model)\n")

# Save the full dataset with all variables (for exploration)
saveRDS(data_with_dummies, "poc_full_data_with_flags.rds")
cat("  - Saved: poc_full_data_with_flags.rds (complete dataset for exploration)\n\n")

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

cat("Key Innovation - MNAR Imputation:\n")
cat("  - Clinical variables: Imputed with median/mode (assumes MCAR)\n")
cat("  - Longitudinal variables: Imputed with fixed 'bad' values + missingness flags\n")
cat("  - This allows the model to learn if missing follow-up predicts nonunion\n\n")

cat("Missingness Flags Created:\n")
for (flag in sort(missingness_flags)) {
  cat("  -", flag, "\n")
}
cat("\n")

cat("Output files created:\n")
cat("  - poc_model_data.rds (Full PoC - Clinical + PROMIS + RUST + all flags)\n")
cat("  - poc_model1_data.rds (Baseline PoC - Clinical + PROMIS + PROMIS flags)\n")
cat("  - poc_full_data_with_flags.rds (Complete dataset for exploration)\n\n")

cat("Next steps:\n")
cat("  1. Load data: model_data <- readRDS('poc_model_data.rds')\n")
cat("  2. Split into train/test sets (stratified by Nonunion_Label)\n")
cat("  3. Train ML models (logistic regression, random forest, XGBoost)\n")
cat("  4. Evaluate performance (AUC, sensitivity, specificity, calibration)\n")
cat("  5. Interpret missingness flag coefficients to understand MNAR patterns\n\n")

cat("Good luck with your modeling!\n")
