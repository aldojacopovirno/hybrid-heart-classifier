#' Run Complete Analysis Pipeline
#' 
#' @description
#' Executes the complete analysis workflow for heart disease data.
#' 
#' @param data_path Path to the CSV file containing heart disease data
#' @return A list containing:
#'   \item{Preprocessed_Data}{Clean and encoded dataset}
#'   \item{Statistical_Analysis}{Statistical analysis results}
#'   \item{Normality_Tests}{Results of normality tests}
#'   \item{Model_Results}{Ordinal regression results}
#'   \item{ROC_Results}{ROC analysis results}
#' @details
#' Pipeline steps:
#' 1. Data preprocessing and cleaning
#' 2. Statistical analysis
#' 3. Normality testing
#' 4. Distribution visualization
#' 5. Ordinal regression analysis
#' 6. ROC curve analysis
run_complete_analysis <- function(data_path) {
  # Load required libraries
  required_packages <- c("readr", "labstatR", "tseries", "moments", "VIM", 
                         "gridExtra", "lmtest", "nortest", "MASS", "car", 
                         "olsrr", "pscl", "pwr", "dplyr", "caret", "pROC")
  
  for(pkg in required_packages) {
    if(!require(pkg, character.only = TRUE)) {
      install.packages(pkg)
      library(pkg, character.only = TRUE)
    }
  }
  
  # Read and prepare data
  datamatrix <- read_csv(data_path)
  cat("Initial data loaded. Starting analysis pipeline...\n")
  
  # Step 1: Data Preprocessing
  cat("\n=== Step 1: Data Preprocessing ===\n")
  df_encoded <- encoder(datamatrix)
  df_imputed <- handle_missing_values(df_encoded)
  df_prepared <- prepare_data(df_imputed)
  
  # Step 2: Statistical Analysis
  cat("\n=== Step 2: Basic Statistical Analysis ===\n")
  stats_results <- statistical_analysis(df_imputed)
  print(stats_results)
  
  # Step 3: Normality Tests
  cat("\n=== Step 3: Normality Tests ===\n")
  normality_results <- normality_tests_selected(df_imputed)
  print(normality_results)
  
  # Step 4: Distribution Visualization
  cat("\n=== Step 4: Generating Visualizations ===\n")
  par(mfrow = c(2, 2))
  generate_distribution_plots(df_imputed)
  cat("Distribution plots generated.\n")
  
  generate_boxplots(df_imputed)
  cat("Boxplots generated.\n")
  
  generate_qqplots(df_imputed)
  cat("Q-Q plots generated.\n")
  par(mfrow = c(1, 1))
  
  # Step 5: Ordinal Regression Analysis
  cat("\n=== Step 5: Ordinal Regression Analysis ===\n")
  predictors <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalch", "exang", "oldpeak", "slope", "ca", "thal")
  
  # Create formula for the model
  formula <- as.formula(paste("num ~", paste(predictors, collapse = " + ")))
  
  # Fit ordinal regression model
  model <- polr(formula, data = df_prepared, Hess = TRUE)
  
  # Calculate model metrics
  model_summary <- summary(model)
  vif_values <- vif(model)
  aic_value <- AIC(model)
  bic_value <- BIC(model)
  mcfadden_r2 <- pR2(model)["McFadden"]  # Extract just the McFadden value
  
  # Calculate predicted probabilities and classes
  predicted_probs <- predict(model, type = "probs")
  predicted_classes <- apply(predicted_probs, 1, which.max)
  
  # Create confusion matrix
  confusion_matrix <- table(Actual = df_prepared$num, 
                            Predicted = factor(predicted_classes, 
                                               levels = levels(df_prepared$num)))
  
  # Calculate accuracy
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  # Calculate class-wise metrics
  precision <- diag(confusion_matrix) / rowSums(confusion_matrix)
  recall <- diag(confusion_matrix) / colSums(confusion_matrix)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  evaluation_metrics <- data.frame(
    Class = levels(df_prepared$num),
    Precision = round(precision, 4),
    Recall = round(recall, 4),
    F1_Score = round(f1_score, 4)
  )
  
  # Store model results
  model_results <- list(
    Model = model,
    Model_Summary = model_summary,
    VIF_Values = vif_values,
    AIC = aic_value,
    BIC = bic_value,
    McFadden_R2 = mcfadden_r2,
    Confusion_Matrix = confusion_matrix,
    Accuracy = accuracy,
    Evaluation_Metrics = evaluation_metrics
  )
  
  # Print model results
  cat("\nModel Summary:\n")
  print(model_results$Model_Summary)
  
  cat("\nVIF Values:\n")
  print(model_results$VIF_Values)
  
  cat("\nModel Information Criteria:\n")
  cat("AIC:", model_results$AIC, "\n")
  cat("BIC:", model_results$BIC, "\n")
  cat("McFadden R²:", model_results$McFadden_R2, "\n") 
  
  cat("\nConfusion Matrix:\n")
  print(model_results$Confusion_Matrix)
  
  cat("\nModel Accuracy:", model_results$Accuracy, "\n")
  
  cat("\nEvaluation Metrics by Class:\n")
  print(model_results$Evaluation_Metrics)
  
  # Step 6: ROC Analysis
  cat("\n=== Step 6: ROC Analysis ===\n")
  roc_results <- generate_roc_curves(model_results$Model, df_prepared, "num")
  
  cat("\nAUC Values for each class:\n")
  print(data.frame(
    Class = 1:length(roc_results$AUC_Values),
    AUC = round(roc_results$AUC_Values, 3)
  ))
  
  # Return all results in a structured list
  return(list(
    Preprocessed_Data = df_prepared,
    Statistical_Analysis = stats_results,
    Normality_Tests = normality_results,
    Model_Results = model_results,
    ROC_Results = roc_results
  ))
}
