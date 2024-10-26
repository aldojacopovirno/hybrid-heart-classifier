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
         xlab = "False Positive Rate", 
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