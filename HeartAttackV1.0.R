setwd('/Users/aldojacopo/Library/CloudStorage/OneDrive-Uniparthenope/SIAFA 2 - Analisi Espolarativa/ProjectWorkR')

# Load necessary libraries
library(readr)
library(labstatR)
library(tseries)
library(moments)
library(VIM)
library(ggplot2)
library(gridExtra)

# Load the dataset
df <- read_csv("heart_disease_uci.csv")
cat("Structure of the dataset:\n")
str(df)
cat("\nFirst few rows of the dataset:\n")
print(head(df))

# Function to calculate statistical summary for numeric columns
stat_summary <- function(df) {
  
  numeric_cols <- df[, sapply(df, is.numeric)]
  
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

# Function to check for missing values in the dataset
check_missing_values <- function(df) {
  
  missing_values <- sapply(df, function(col) sum(is.na(col)))
  
  if (sum(missing_values) == 0) {
    cat("\n✔ Non ci sono valori mancanti nel dataframe.\n")
  } else {
    cat("\n⚠ Ci sono valori mancanti nel dataframe:\n")
    print(missing_values[missing_values > 0])
  }
}

# Convert categorical columns to factors
df$ca <- as.factor(df$ca)
df$num <- as.factor(df$num)

# Function for KNN imputation of missing values
impute_knn <- function(df, k = 5) {
  if (any(is.na(df))) {
    imputed_df <- kNN(df, k = k, imp_var = FALSE)
    cat("\n Imputazione dei valori NA completata.\n")
    return(imputed_df)
  } else {
    message("\n Non ci sono valori NA nel dataframe.")
    return(df)
  }
}

# Impute missing values using KNN
imputed_df <- impute_knn(df, k = 3)

# Function to generate distribution plots with normal curve
plot_distribution_with_stats <- function(df, col) {
  
  mean_val <- mean(df[[col]], na.rm = TRUE)
  median_val <- median(df[[col]], na.rm = TRUE)
  
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

# Function to generate individual plots for quantitative variables
plot_individual_quantitative <- function(df) {
  
  numeric_cols <- df[, sapply(df, is.numeric) & !(names(df) %in% c("oldpeak", "id")), drop = FALSE]
  
  for (col in names(numeric_cols)) {
    cat("\n Generazione grafico per variabile quantitativa:", col, "\n")
    
    # Distribution plot with histogram and normal density
    p_dist <- plot_distribution_with_stats(df, col)
    print(p_dist)  # Show distribution plot
    
    # Histogram plot
    p_hist <- ggplot(df, aes_string(x = col)) +
      geom_histogram(binwidth = 30, fill = "blue", color = "black") +
      ggtitle(paste("Istogramma di", col)) +
      theme_minimal()
    print(p_hist)  # Show histogram plot
    
    # Boxplot
    p_box <- ggplot(df, aes_string(y = col)) +
      geom_boxplot(fill = "orange", color = "black") +
      ggtitle(paste("Boxplot di", col)) +
      theme_minimal()
    print(p_box)  # Show boxplot
  }
}

# Function to generate individual plots for categorical variables
plot_individual_categorical <- function(df) {
  
  categorical_cols <- df[, sapply(df, is.factor) | sapply(df, is.character), drop = FALSE]
  
  for (col in names(categorical_cols)) {
    cat("\n Generazione grafico per variabile categorica:", col, "\n")
    
    categorical_plot <- ggplot(df, aes_string(x = col)) +
      geom_bar(fill = "steelblue", color = "black") +
      ggtitle(paste("Distribuzione di", col)) +
      theme_minimal()
    
    print(categorical_plot)  # Show categorical plot
  }
}

# Generate plots for quantitative variables
cat("\n Generazione dei grafici per le variabili quantitative:\n")
plot_individual_quantitative(imputed_df)

# Generate plots for categorical variables
cat("\n Generazione dei grafici per le variabili categoriche:\n")
plot_individual_categorical(imputed_df)

cat("\n Analisi completata con successo!\n")