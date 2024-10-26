#' Generate Distribution Plots
#' 
#' @description
#' Creates histograms with overlaid normal curves for selected variables.
#' 
#' @param df A data frame containing the variables to plot
#' @return NULL (generates plots as side effect)
#' @details
#' Creates a 2x2 grid of histograms for:
#' - age
#' - trestbps (blood pressure)
#' - chol (cholesterol)
#' - thalch (maximum heart rate)
generate_distribution_plots <- function(df) {
  selected_vars <- c("age", "trestbps", "chol", "thalch")
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  
  for (var in selected_vars) {
    mu <- mean(df[[var]], na.rm = TRUE)
    sigma <- sd(df[[var]], na.rm = TRUE)
    
    hist(df[[var]], 
         main = paste("Histogram of", var), 
         xlab = var, 
         ylab = "Frequency", 
         probability = TRUE,
         col = "lightblue", 
         border = "black")
    
    curve(dnorm(x, mean = mu, sd = sigma), 
          add = TRUE, 
          col = "red", 
          lwd = 2)
  }
  
  par(mfrow = c(1, 1))
}

#' Generate Boxplots
#' 
#' @description
#' Creates boxplots for selected continuous variables.
#' 
#' @param df A data frame containing the variables to plot
#' @return NULL (generates plots as side effect)
#' @details
#' Creates notched boxplots showing:
#' - Median
#' - Quartiles
#' - Outliers
generate_boxplots <- function(df) {
  selected_vars <- c("age", "trestbps", "chol", "thalch")
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  
  for (var in selected_vars) {
    boxplot(df[[var]], 
            main = paste("Boxplot of", var), 
            ylab = var, 
            col = "lightblue", 
            border = "black", 
            notch = TRUE)
  }
  par(mfrow = c(1, 1))
}

#' Generate Q-Q Plots
#' 
#' @description
#' Creates quantile-quantile plots for assessing normality.
#' 
#' @param df A data frame containing the variables to plot
#' @return NULL (generates plots as side effect)
#' @details
#' Creates Q-Q plots with reference lines for visual assessment of normality
generate_qqplots <- function(df) {
  selected_vars <- c("age", "trestbps", "chol", "thalch")
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  
  for (var in selected_vars) {
    qqnorm(df[[var]], main = paste("Q-Q Plot for", var), 
           xlab = "Theoretical Quantiles", 
           ylab = "Sample Quantiles")
    qqline(df[[var]], col = "red", lwd = 2)
  }
  
  par(mfrow = c(1, 1))
}
