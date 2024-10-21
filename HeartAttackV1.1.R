# Set the working directory to where the dataset is stored
# This ensures that the subsequent read_csv function can access the dataset correctly.
setwd("~/Code Nest/PWEDA-CodeNest")

# Load necessary libraries for data manipulation, statistical analysis, imputation, and visualization.
library(readr)      # For reading CSV files
library(labstatR)   # For statistical functions (e.g., skewness, kurtosis)
library(tseries)    # For additional time series and statistical analysis
library(moments)    # For calculating skewness and kurtosis
library(VIM)        # For handling missing data (KNN imputation)
library(ggplot2)    # For data visualization
library(gridExtra)  # For arranging multiple plots in a grid

# Load the dataset from a CSV file
df <- read_csv("heart_disease_uci.csv")

# Print the structure of the dataset to show column names, types, and overall structure
cat("Structure of the dataset:\n")
str(df)

# Print the first few rows of the dataset to get a quick overview
cat("\nFirst few rows of the dataset:\n")
print(head(df))

#' @title stat_summary
#' @description Calculate statistical summary for numeric columns in the dataset.
#' @param df A data frame containing the dataset.
#' @return A data frame with summary statistics (min, max, mean, median, SD, variance, IQR, skewness, kurtosis) for each numeric column.
stat_summary <- function(df) {
  
  # Select only numeric columns from the dataset
  numeric_cols <- df[, sapply(df, is.numeric)]
  
  # Create a summary table for the selected numeric columns
  summary_stats <- data.frame(
    Min = sapply(numeric_cols, min, na.rm = TRUE),
    Max = sapply(numeric_cols, max, na.rm = TRUE),
    Mean = sapply(numeric_cols, mean, na.rm = TRUE),
    Median = sapply(numeric_cols, median, na.rm = TRUE),
    SD = sapply(numeric_cols, sd, na.rm = TRUE),
    Variance = sapply(numeric_cols, var, na.rm = TRUE),
    IQR = sapply(numeric_cols, IQR, na.rm = TRUE),
    Skewness = sapply(numeric_cols, skewness, na.rm = TRUE),
    Kurtosis = sapply(numeric_cols, kurtosis, na.rm = TRUE)
  )
  
  return(summary_stats)
}

#' @title check_missing_values
#' @description Check for missing values in the dataset and display a summary.
#' @param df A data frame containing the dataset.
#' @return Prints a message indicating if there are missing values or not, along with a summary of missing values.
check_missing_values <- function(df) {
  
  # Calculate the number of missing values per column
  missing_values <- sapply(df, function(col) sum(is.na(col)))
  
  # If there are no missing values, print a confirmation message; otherwise, display the columns with missing values
  if (sum(missing_values) == 0) {
    cat("\n✔ Non ci sono valori mancanti nel dataframe.\n")
  } else {
    cat("\n⚠ Ci sono valori mancanti nel dataframe:\n")
    print(missing_values[missing_values > 0])
  }
}

# Convert certain categorical columns to factor types for appropriate statistical analysis
df$ca <- as.factor(df$ca)
df$num <- as.factor(df$num)

#' @title impute_knn
#' @description Perform KNN imputation for missing values in the dataset.
#' @param df A data frame with potential missing values.
#' @param k The number of nearest neighbors to use for KNN imputation (default is 5).
#' @return A data frame with imputed values where missing values were found.
impute_knn <- function(df, k = 5) {
  
  # Check if there are any missing values in the dataset
  if (any(is.na(df))) {
    
    # Perform KNN imputation for missing values
    imputed_df <- kNN(df, k = k, imp_var = FALSE)
    cat("\n Imputazione dei valori NA completata.\n")
    return(imputed_df)
  } else {
    message("\n Non ci sono valori NA nel dataframe.")
    return(df)
  }
}

# Impute missing values using KNN with k = 3 (3 nearest neighbors)
imputed_df <- impute_knn(df, k = 3)

#' @title plot_distribution_with_stats
#' @description Plot the distribution of a numeric column, including the normal curve and vertical lines for the mean and median.
#' @param df A data frame containing the dataset.
#' @param col The name of the numeric column to be plotted.
#' @return A ggplot object with the histogram, normal curve, and vertical lines for mean and median.
plot_distribution_with_stats <- function(df, col) {
  
  # Calculate mean and median for the specified column
  mean_val <- mean(df[[col]], na.rm = TRUE)
  median_val <- median(df[[col]], na.rm = TRUE)
  
  # Create a ggplot object for the distribution plot
  p <- ggplot(df, aes_string(x = col)) +
    geom_histogram(aes(y = ..density..), binwidth = 30, fill = "lightblue", color = "black", alpha = 0.7) +
    stat_function(fun = dnorm, 
                  args = list(mean = mean_val, sd = sd(df[[col]], na.rm = TRUE)),
                  color = "red", size = 1) +
    geom_vline(aes(xintercept = mean_val), color = "blue", linetype = "dashed", size = 1) +
    geom_vline(aes(xintercept = median_val), color = "green", linetype = "dashed", size = 1) +
    ggtitle(paste("Distribuzione di", col, ": Media e Mediana")) +
    theme_minimal() +
    xlab(col) +
    ylab("Densità") +
    annotate("text", x = mean_val, y = 0.02, label = "Media", color = "blue", angle = 90, vjust = -0.5) +
    annotate("text", x = median_val, y = 0.02, label = "Mediana", color = "green", angle = 90, vjust = -0.5)
  
  return(p)
}

#' @title plot_individual_quantitative
#' @description Generate and display distribution plots for each quantitative variable in the dataset.
#' @param df A data frame containing the dataset.
#' @return Prints a distribution plot, histogram, and boxplot for each numeric column in the dataset.
plot_individual_quantitative <- function(df) {
  
  # Select numeric columns for plotting (excluding some irrelevant columns)
  numeric_cols <- df[, sapply(df, is.numeric) & !(names(df) %in% c("oldpeak", "id")), drop = FALSE]
  
  # Loop through each numeric column and generate the plots
  for (col in names(numeric_cols)) {
    cat("\n Generazione grafico per variabile quantitativa:", col, "\n")
    
    # Generate and print the distribution plot with stats
    p_dist <- plot_distribution_with_stats(df, col)
    print(p_dist)
    
    # Generate and print a simple histogram
    p_hist <- ggplot(df, aes_string(x = col)) +
      geom_histogram(binwidth = 30, fill = "blue", color = "black") +
      ggtitle(paste("Istogramma di", col)) +
      theme_minimal()
    print(p_hist)
    
    # Generate and print a boxplot
    p_box <- ggplot(df, aes_string(y = col)) +
      geom_boxplot(fill = "orange", color = "black") +
      ggtitle(paste("Boxplot di", col)) +
      theme_minimal()
    print(p_box)
  }
}

#' @title plot_individual_categorical
#' @description Generate and display bar plots for each categorical variable in the dataset.
#' @param df A data frame containing the dataset.
#' @return Prints a bar plot for each categorical column in the dataset.
plot_individual_categorical <- function(df) {
  
  # Select categorical columns for plotting
  categorical_cols <- df[, sapply(df, is.factor) | sapply(df, is.character), drop = FALSE]
  
  # Loop through each categorical column and generate the plots
  for (col in names(categorical_cols)) {
    cat("\n Generazione grafico per variabile categorica:", col, "\n")
    
    # Generate and print a bar plot for the categorical variable
    categorical_plot <- ggplot(df, aes_string(x = col)) +
      geom_bar(fill = "steelblue", color = "black") +
      ggtitle(paste("Distribuzione di", col)) +
      theme_minimal()
    
    print(categorical_plot)
  }
}

# Generate plots for quantitative variables
cat("\n Generazione dei grafici per le variabili quantitative:\n")
plot_individual_quantitative(imputed_df)

# Generate plots for categorical variables
cat("\n Generazione dei grafici per le variabili categoriche:\n")
plot_individual_categorical(imputed_df)

# Final confirmation message
cat("\n Analisi completata con successo!\n")