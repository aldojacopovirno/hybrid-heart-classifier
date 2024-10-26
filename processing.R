#' Encode Categorical Variables
#' 
#' @description
#' Converts categorical variables in the heart disease dataset to numeric format.
#' 
#' @param df A data frame containing heart disease data
#' @return A data frame with encoded categorical variables
#' @details
#' Performs the following encodings:
#' - sex: Male = 1, Female = 0
#' - cp: Chest pain types (0-3)
#' - restecg: ECG results (0-2)
#' - slope: ST segment slope (0-2)
#' - thal: Thalassemia types (0-2)
encoder <- function(df) {
  transformed_data <- df
  transformed_data$sex <- ifelse(transformed_data$sex == "Male", 1, 0)
  transformed_data$cp <- factor(transformed_data$cp, 
                                levels = c("typical angina", "atypical angina", 
                                           "non-anginal", "asymptomatic"))
  transformed_data$cp <- as.numeric(transformed_data$cp) - 1
  transformed_data$restecg <- factor(transformed_data$restecg, 
                                     levels = c("normal", "st-t wave abnormality", 
                                                "lv hypertrophy"))
  transformed_data$restecg <- as.numeric(transformed_data$restecg) - 1
  transformed_data$fbs <- as.numeric(transformed_data$fbs)
  transformed_data$exang <- as.numeric(transformed_data$exang)
  transformed_data$slope <- factor(transformed_data$slope, 
                                   levels = c("upsloping", "flat", "downsloping"))
  transformed_data$slope <- as.numeric(transformed_data$slope) - 1
  transformed_data$thal <- factor(transformed_data$thal, 
                                  levels = c("normal", "fixed defect", 
                                             "reversable defect"))
  transformed_data$thal <- as.numeric(transformed_data$thal) - 1
  transformed_data$ca <- as.numeric(as.character(transformed_data$ca))
  transformed_data$ca[is.na(transformed_data$ca)] <- 
    median(transformed_data$ca, na.rm = TRUE)
  return(transformed_data)
}

#' Handle Missing Values Using KNN Imputation
#' 
#' @description
#' Identifies and imputes missing values in the dataset using K-Nearest Neighbors.
#' 
#' @param df A data frame with potential missing values
#' @return A data frame with imputed values
#' @details
#' - Reports the count of missing values per column
#' - Uses KNN imputation with k=5 if missing values are found
handle_missing_values <- function(df) {
  missing_values <- sapply(df, function(x) sum(is.na(x)))
  cat("Valori mancanti per colonna:\n")
  print(missing_values[missing_values > 0])
  if (sum(missing_values) > 0) {
    imputed_data <- kNN(df, k = 5, imp_var = FALSE)
    return(imputed_data)
  } else {
    cat("Nessun valore mancante trovato nel dataset.\n")
    return(df)
  }
}

#' Prepare Data for Ordinal Regression
#' 
#' @description
#' Prepares the dataset for ordinal regression analysis.
#' 
#' @param df A data frame to prepare
#' @return A data frame with properly formatted response variable
#' @details
#' Converts the target variable 'num' to an ordered factor
prepare_data <- function(df) {
  # Convert response variable to ordered factor
  df$num <- factor(df$num, ordered = TRUE, 
                   levels = sort(unique(df$num)))
  return(df)
}