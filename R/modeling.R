#' Generate ROC Curves for Ordinal Model
#' 
#' @description
#' Creates ROC curves for multiclass classification results.
#' 
#' @param model A fitted ordinal regression model
#' @param data The dataset used for modeling
#' @param response_var Name of the response variable
#' @return A list containing:
#'   \item{ROC_Objects}{ROC curve objects for each class}
#'   \item{AUC_Values}{Area Under Curve values for each class}
#' @details
#' - Generates one ROC curve per class
#' - Calculates AUC values
#' - Creates a plot with all curves and a legend
generate_roc_curves <- function(model, data, response_var) {
  require(pROC)
  require(ggplot2)
  
  # Suppress messages and warnings for ROC calculations
  suppressMessages({
    # Get predicted probabilities
    pred_probs <- predict(model, type = "probs")
    # Convert response variable to numeric for ROC analysis
    actual_values <- as.numeric(data[[response_var]])
    # Initialize lists to store ROC objects and AUC values
    roc_list <- list()
    auc_values <- numeric()
    # Calculate ROC curve for each class
    n_classes <- ncol(pred_probs)
    
    # Create plot
    par(mfrow = c(1, 1))
    plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1),
         xlab = "Sensitivity", 
         ylab = "True Positive Rate",
         main = "ROC Curves for Each Class",
         asp = 1)
    
    # Add diagonal reference line
    abline(0, 1, lty = 2, col = "gray")
    
    # Colors for different classes
    colors <- rainbow(n_classes)
    
    # Calculate and plot ROC curve for each class
    for(i in 1:n_classes) {
      # Create binary classification for current class
      binary_actual <- ifelse(actual_values == i, 1, 0)
      # Calculate ROC curve with message suppression
      roc_obj <- roc(binary_actual, pred_probs[,i], quiet = TRUE)
      roc_list[[i]] <- roc_obj
      auc_values[i] <- auc(roc_obj)
      # Plot ROC curve
      lines(1 - roc_obj$specificities, roc_obj$sensitivities, 
            col = colors[i], lwd = 2)
    }
    
    # Add legend
    legend("bottomright", 
           legend = paste("Class", 1:n_classes, 
                          " (AUC =", round(auc_values, 3), ")"),
           col = colors, 
           lwd = 2,
           cex = 0.8)
  })
  
  # Return AUC values
  return(list(
    ROC_Objects = roc_list,
    AUC_Values = auc_values
  ))
}

#' Run Statistical Analysis
#' 
#' @description
#' Executes a pipeline for statistical analysis including data loading, preprocessing,
#' normality tests, and visualization.
#' 
#' @param data_path The file path to the dataset in CSV format.
#' @return A list containing:
#'   \item{Preprocessed_Data}{The dataset after preprocessing steps.}
#'   \item{Statistical_Analysis}{Results of basic statistical analysis.}
#'   \item{Normality_Tests}{Results of normality tests conducted on the data.}
#' @details
#' - Loads necessary libraries if not already installed.
#' - Encodes categorical variables and scales specified numerical columns.
#' - Generates visualizations including distribution plots, boxplots, and Q-Q plots.
run_statistical_analysis <- function(data_path) {
  # Load required libraries
  required_packages <- c("readr", "labstatR", "tseries", "moments", "VIM", 
                         "gridExtra", "lmtest", "nortest", "MASS", "car", 
                         "olsrr", "pscl", "pwr", "dplyr", "caret", "pROC", "brant")
  
  for(pkg in required_packages) {
    if(!require(pkg, character.only = TRUE)) {
      install.packages(pkg)
      library(pkg, character.only = TRUE)
    }
  }
  
  # Read and prepare data
  datamatrix <- read_csv(data_path)
  cat("Initial data loaded. Starting statistical analysis pipeline...\n")
  
  # Step 1: Data Preprocessing
  cat("\n=== Step 1: Data Preprocessing ===\n")
  df_encoded <- encoder(datamatrix)
  
  # Specify the columns to scale
  columns_to_scale <- c("age", "trestbps", "chol", "thalch")
  
  # Apply the robust scaler with the specified columns
  df_scaled <- apply_robust_scaler(df_encoded, columns = columns_to_scale)
  df_imputed <- handle_missing_values(df_encoded)
  
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
  
  # Return results
  return(list(
    Preprocessed_Data = df_imputed,
    Statistical_Analysis = stats_results,
    Normality_Tests = normality_results
  ))
}

#' Ordinal Regression Model
#' 
#' @description
#' Fits an ordinal regression model to the prepared data and evaluates its performance.
#' 
#' @param df_prepared A prepared dataframe for modeling, containing response and predictor variables.
#' @return A list containing:
#'   \item{Model}{The fitted ordinal regression model.}
#'   \item{Model_Summary}{Summary statistics of the model.}
#'   \item{VIF_Values}{Variance Inflation Factor values for predictor variables.}
#'   \item{AIC}{Akaike Information Criterion value.}
#'   \item{BIC}{Bayesian Information Criterion value.}
#'   \item{McFadden_R2}{McFadden's pseudo R-squared value.}
#'   \item{Confusion_Matrix}{Confusion matrix comparing actual vs. predicted classes.}
#'   \item{Accuracy}{Overall accuracy of the model.}
#'   \item{Evaluation_Metrics}{Class-wise evaluation metrics including precision, recall, and F1 score.}
#' @details
#' - Assumes the response variable 'num' is a factor.
#' - Conducts a Brant test for proportional odds assumption.
#' - Generates predicted probabilities and classes for model evaluation.
ordinal_regression_model <- function(df_prepared) {
  # Load required libraries
  required_packages <- c("MASS", "car", "pscl", "pROC", "brant")
  
  for(pkg in required_packages) {
    if(!require(pkg, character.only = TRUE)) {
      install.packages(pkg)
      library(pkg, character.only = TRUE)
    }
  }
  
  # Ensure 'num' is a factor
  df_prepared$num <- as.factor(df_prepared$num)
  
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
  
  # Brant Test for Proportional Odds Assumption
  cat("\n=== Step 5a: Brant Test for Proportional Odds Assumption ===\n")
  brant_test <- brant(model)
  print(brant_test)
  
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
    Model_Results = model_results,
    ROC_Results = roc_results
  ))
}


#' Train the XGBoost Model
#' 
#' @description
#' Trains an XGBoost model on the provided dataset with specified response and predictor variables.
#' 
#' @param df The dataframe containing the dataset.
#' @param response_var The name of the response variable as a string.
#' @param predictor_vars A vector of predictor variable names.
#' @param n_trees The number of trees to use in the model (default is 100).
#' @param learning_rate The learning rate for the model (default is 0.1).
#' @return A trained XGBoost model.
#' @details
#' - The response variable should be a factor.
#' - Uses 80% of the data for training and 20% for testing.
#' - Outputs the model performance metrics.
xgb_model <- function(df, response_var, predictor_vars, 
                      n_trees = 100, learning_rate = 0.1) {
  # Set seed for reproducibility
  set.seed(123)
  
  # Split data into training (80%) and testing (20%) sets
  train_index <- createDataPartition(df[[response_var]], p = .8, 
                                     list = FALSE, times = 1)
  train_data <- df[train_index, ]
  test_data <- df[-train_index, ]
  
  # Prepare data matrices for XGBoost
  dtrain <- xgb.DMatrix(data = as.matrix(train_data[predictor_vars]), 
                        label = as.numeric(train_data[[response_var]]) - 1)
  
  # Define model parameters
  params <- list(
    objective = "multi:softprob",    # Multiclass probability output
    num_class = length(levels(df[[response_var]])),
    eval_metric = c("mlogloss", "merror"),  # Added error rate metric
    eta = learning_rate,             # Learning rate
    max_depth = 3,                   # Maximum tree depth
    nthread = 2                      # Number of parallel threads
  )
  
  # Train the model
  xgb_model <- xgb.train(params, dtrain, nrounds = n_trees)
  
  # Make predictions on test set
  dtest <- xgb.DMatrix(data = as.matrix(test_data[predictor_vars]))
  test_predictions <- predict(xgb_model, dtest)
  
  # Convert predictions to class probabilities and labels
  pred_probs <- matrix(test_predictions, 
                       ncol = length(levels(df[[response_var]])), 
                       byrow = TRUE)
  predicted_classes <- max.col(pred_probs) - 1
  
  # Prepare factors for confusion matrix
  predicted_classes_factor <- factor(predicted_classes, 
                                     levels = 0:(length(levels(df[[response_var]])) - 1))
  actual_classes_factor <- factor(as.numeric(test_data[[response_var]]) - 1, 
                                  levels = 0:(length(levels(df[[response_var]])) - 1))
  
  # Calculate confusion matrix and basic metrics
  confusion_mat <- confusionMatrix(predicted_classes_factor, actual_classes_factor)
  
  # Calculate class-specific metrics
  class_metrics <- data.frame(
    Class = levels(df[[response_var]]),
    Precision = confusion_mat$byClass[, "Precision"],
    Recall = confusion_mat$byClass[, "Sensitivity"],
    F1_Score = confusion_mat$byClass[, "F1"],
    Specificity = confusion_mat$byClass[, "Specificity"]
  )
  
  # Calculate macro-averaged metrics
  macro_metrics <- colMeans(class_metrics[, -1], na.rm = TRUE)
  
  # Calculate ROC curves and AUC for each class
  roc_curves <- list()
  auc_scores <- numeric(length(levels(df[[response_var]])))
  
  for(i in 1:length(levels(df[[response_var]]))) {
    actual_binary <- as.numeric(actual_classes_factor == (i-1))
    pred_prob <- pred_probs[, i]
    roc_obj <- roc(actual_binary, pred_prob)
    roc_curves[[i]] <- roc_obj
    auc_scores[i] <- auc(roc_obj)
  }
  
  # Print comprehensive model evaluation
  cat("\nModel Performance Metrics:\n")
  cat("=========================\n\n")
  
  # Overall metrics
  cat("Overall Metrics:\n")
  cat("---------------\n")
  cat(sprintf("Accuracy: %.3f\n", confusion_mat$overall['Accuracy']))
  cat(sprintf("Kappa: %.3f\n", confusion_mat$overall['Kappa']))
  cat(sprintf("Precision: %.3f\n", macro_metrics['Precision']))
  cat(sprintf("Recall: %.3f\n", macro_metrics['Recall']))
  cat(sprintf("F1 Score: %.3f\n", macro_metrics['F1_Score']))
  cat(sprintf("Mean ROC-AUC: %.3f\n", mean(auc_scores)))
  
  # Class-specific metrics
  cat("\nClass-specific Metrics:\n")
  cat("---------------------\n")
  print(class_metrics)
  
  # Feature importance
  importance_matrix <- xgb.importance(feature_names = predictor_vars, model = xgb_model)
  cat("\nFeature Importance:\n")
  cat("------------------\n")
  print(importance_matrix)
  
  # 1. Feature importance plot
  xgb.plot.importance(importance_matrix,
                      main = "Importanza delle Feature nel Modello XGBoost",
                      xlab = "Importanza",
                      col = "steelblue",  # Colore delle barre
                      cex.axis = 0.8,     # Dimensione del testo sugli assi
                      cex.lab = 0.9)      # Dimensione delle etichette
  
  # 2. ROC curves
  plot(roc_curves[[1]], main = "ROC Curves by Class")
  for(i in 2:length(roc_curves)) {
    lines(roc_curves[[i]], col = i)
  }
  legend("bottomright", 
         legend = paste("Class", levels(df[[response_var]]), 
                        sprintf("(AUC = %.3f)", auc_scores)),
         col = 1:length(roc_curves), 
         lwd = 2)
  
  # Return model and evaluation results
  return(list(
    model = xgb_model,
    confusion_matrix = confusion_mat,
    class_metrics = class_metrics,
    macro_metrics = macro_metrics,
    roc_curves = roc_curves,
    auc_scores = auc_scores,
    feature_importance = importance_matrix
  ))
}
