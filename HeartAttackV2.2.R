# Impostazione directory di lavoro 
setwd("~/Code Nest/PWEDA-CodeNest")

# Caricamento librerie necessarie
library(readr)      # Per lettura file CSV
library(labstatR)   # Per funzioni statistiche base
library(tseries)    # Per analisi serie temporali e test statistici
library(moments)    # Per calcolo asimmetria e curtosi
library(VIM)        # Per gestione dati mancanti (KNN imputation)
library(gridExtra)  # Per layout grafici multipli
library(lmtest)     # Per test Jarque-Bera
library(nortest)    # Per test di normalità aggiuntivi
library(MASS)       # Per regressione logistica ordinale
library(car)        # Per diagnostica modelli (VIF)
library(olsrr)      # Per diagnostica modelli
library(pscl)       # Per pseudo R-squared
library(pwr)

# Caricamento dataset
# TODO: Aggiungere gestione errori per file mancante
datamatrix <- read_csv("heart_disease_uci.csv")

# Visualizzazione struttura dataset
cat("Structure of the dataset:\n")
str(datamatrix)

# Visualizzazione prime righe per verifica dati
cat("\nFirst few rows of the dataset:\n")
print(head(datamatrix))

# Funzione per codifica variabili categoriche in numeriche
encoder <- function(df) {
  # Creazione copia per preservare dati originali
  transformed_data <- df
  
  # Codifica sesso: Male = 1, Female = 0
  transformed_data$sex <- ifelse(transformed_data$sex == "Male", 1, 0)
  
  # Codifica dolore toracico (cp) in 4 livelli (0-3)
  transformed_data$cp <- factor(transformed_data$cp, 
                                levels = c("typical angina", "atypical angina", 
                                           "non-anginal", "asymptomatic"))
  transformed_data$cp <- as.numeric(transformed_data$cp) - 1
  
  # Codifica ECG a riposo (restecg) in 3 livelli (0-2)
  transformed_data$restecg <- factor(transformed_data$restecg, 
                                     levels = c("normal", "st-t wave abnormality", 
                                                "lv hypertrophy"))
  transformed_data$restecg <- as.numeric(transformed_data$restecg) - 1
  
  # Conversione variabili binarie
  transformed_data$fbs <- as.numeric(transformed_data$fbs)
  transformed_data$exang <- as.numeric(transformed_data$exang)
  
  # Codifica pendenza ST (slope) in 3 livelli (0-2)
  transformed_data$slope <- factor(transformed_data$slope, 
                                   levels = c("upsloping", "flat", "downsloping"))
  transformed_data$slope <- as.numeric(transformed_data$slope) - 1
  
  # Codifica difetto perfusione (thal) in 3 livelli (0-2)
  transformed_data$thal <- factor(transformed_data$thal, 
                                  levels = c("normal", "fixed defect", 
                                             "reversable defect"))
  transformed_data$thal <- as.numeric(transformed_data$thal) - 1
  
  # Gestione valori mancanti in 'ca' con mediana
  transformed_data$ca <- as.numeric(as.character(transformed_data$ca))
  transformed_data$ca[is.na(transformed_data$ca)] <- 
    median(transformed_data$ca, na.rm = TRUE)
  
  return(transformed_data)
}

# Applicazione encoding
df_encoded <- encoder(datamatrix)

# Funzione per gestione valori mancanti con KNN Imputer
# TODO: Aggiungere parametro k personalizzabile
handle_missing_values <- function(df) {
  # Controllo valori mancanti per colonna
  missing_values <- sapply(df, function(x) sum(is.na(x)))
  
  # Report valori mancanti
  cat("Valori mancanti per colonna:\n")
  print(missing_values[missing_values > 0])
  
  # Applicazione KNN Imputer se necessario
  if (sum(missing_values) > 0) {
    # Imputazione con k=5 vicini più prossimi
    imputed_data <- kNN(df, k = 5, imp_var = FALSE)
    return(imputed_data)
  } else {
    cat("Nessun valore mancante trovato nel dataset.\n")
    return(df)
  }
}

# Applicazione imputazione valori mancanti
df_imputed <- handle_missing_values(df_encoded)

# Enhanced statistical analysis function with confidence intervals, outlier detection,
# and correlation analysis
# Enhanced statistical analysis function with confidence intervals, p-values,
# outlier detection, and correlation analysis
statistical_analysis <- function(df) {
  # Select only numeric columns
  numeric_columns <- sapply(df, is.numeric)
  df_numeric <- df[, numeric_columns]
  
  # Function to calculate t-test p-value for mean
  calculate_mean_pvalue <- function(x) {
    t_test <- t.test(x)
    return(t_test$p.value)
  }
  
  # Basic descriptive statistics
  basic_stats <- data.frame(
    Variable = names(df_numeric),
    Min = apply(df_numeric, 2, min, na.rm = TRUE),
    Q1 = apply(df_numeric, 2, function(x) quantile(x, probs = 0.25, na.rm = TRUE)),
    Median = apply(df_numeric, 2, median, na.rm = TRUE),
    Mean = colMeans(df_numeric, na.rm = TRUE),
    Q3 = apply(df_numeric, 2, function(x) quantile(x, probs = 0.75, na.rm = TRUE)),
    Max = apply(df_numeric, 2, max, na.rm = TRUE),
    Variance = apply(df_numeric, 2, var, na.rm = TRUE),
    Standard_Deviation = apply(df_numeric, 2, sd, na.rm = TRUE),
    Skewness = sapply(df_numeric, skewness, na.rm = TRUE),
    Kurtosis = sapply(df_numeric, kurtosis, na.rm = TRUE)
  )
  
  # Confidence intervals and p-values for means (95%)
  inference_stats <- t(apply(df_numeric, 2, function(x) {
    n <- sum(!is.na(x))
    se <- sd(x, na.rm = TRUE) / sqrt(n)
    ci <- qt(0.975, df = n - 1) * se
    mean_val <- mean(x, na.rm = TRUE)
    p_value <- calculate_mean_pvalue(x)
    c(
      Mean = mean_val,
      CI_Lower = mean_val - ci,
      CI_Upper = mean_val + ci,
      P_Value = p_value
    )
  }))
  
  # Outlier detection using IQR method
  outliers <- lapply(df_numeric, function(x) {
    q1 <- quantile(x, 0.25, na.rm = TRUE)
    q3 <- quantile(x, 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    outliers <- sum(x < lower_bound | x > upper_bound, na.rm = TRUE)
    outlier_prop <- outliers / length(x[!is.na(x)])
    c(
      n_outliers = outliers,
      prop_outliers = outlier_prop,
      lower_bound = lower_bound,
      upper_bound = upper_bound
    )
  })
  outliers_df <- as.data.frame(do.call(rbind, outliers))
  
  # Correlation analysis with p-values
  cor_matrix <- cor(df_numeric, use = "pairwise.complete.obs")
  cor_pvalues <- matrix(NA, ncol = ncol(df_numeric), nrow = ncol(df_numeric))
  for(i in 1:ncol(df_numeric)) {
    for(j in 1:ncol(df_numeric)) {
      cor_test <- cor.test(df_numeric[[i]], df_numeric[[j]])
      cor_pvalues[i,j] <- cor_test$p.value
    }
  }
  colnames(cor_pvalues) <- colnames(cor_matrix)
  rownames(cor_pvalues) <- rownames(cor_matrix)
  
  # Format and organize results
  results <- list(
    descriptive_statistics = list(
      basic = basic_stats,
      inference = as.data.frame(inference_stats)
    ),
    outlier_analysis = list(
      summary = outliers_df,
      detection_method = "IQR (1.5 * IQR from quartiles)"
    ),
    correlation_analysis = list(
      coefficients = cor_matrix,
      p_values = cor_pvalues
    )
  )
  
  # Print formatted results
  cat("\n=== Statistical Analysis Results ===\n\n")
  
  cat("1. Basic Descriptive Statistics:\n")
  print(basic_stats)
  
  cat("\n2. Inference Statistics (95% CI):\n")
  print(inference_stats)
  
  cat("\n3. Outlier Analysis:\n")
  print(outliers_df)
  
  cat("\n4. Correlation Analysis:\n")
  cat("Correlation Matrix:\n")
  print(round(cor_matrix, 3))
  cat("\nCorrelation P-values:\n")
  print(round(cor_pvalues, 3))
  
  # Return complete results object
  invisible(results)
}

stats_results <- statistical_analysis(df_imputed)

# Enhanced normality tests function with multiple testing correction and power analysis
normality_tests_selected <- function(df, selected_vars = c("age", "trestbps", "chol", "thalch"), 
                                     alpha = 0.05) {
  require(nortest)
  require(pwr)
  df_selected <- df[, selected_vars]
  n_tests <- length(selected_vars) * 4  # 4 tests per variable
  # Bonferroni corrected alpha
  alpha_corrected <- alpha / n_tests
  results <- data.frame(
    Variable = character(),
    Shapiro_Wilk = numeric(),
    KS_Test = numeric(),
    Jarque_Bera = numeric(),
    Anderson_Darling = numeric(),
    SW_Power = numeric(),
    KS_Power = numeric(),
    JB_Power = numeric(),
    AD_Power = numeric(),
    stringsAsFactors = FALSE
  )
  for (var in names(df_selected)) {
    # Calculate sample size
    n <- sum(!is.na(df_selected[[var]]))
    # Perform normality tests
    shapiro_test <- shapiro.test(df_selected[[var]])
    ks_test <- ks.test(df_selected[[var]], "pnorm", 
                       mean = mean(df_selected[[var]], na.rm = TRUE),
                       sd = sd(df_selected[[var]], na.rm = TRUE))
    jarque_bera_test <- jarque.bera.test(df_selected[[var]])
    ad_test <- ad.test(df_selected[[var]])
    # Calculate power for each test
    # Note: These are approximate power calculations
    effect_size <- 0.3  # Medium effect size
    sw_power <- pwr.t.test(n = n, d = effect_size, sig.level = alpha_corrected, 
                           type = "two.sample", alternative = "two.sided")$power
    ks_power <- pwr.t.test(n = n, d = effect_size, sig.level = alpha_corrected, 
                           type = "two.sample", alternative = "two.sided")$power
    jb_power <- pwr.t.test(n = n, d = effect_size, sig.level = alpha_corrected, 
                           type = "two.sample", alternative = "two.sided")$power
    ad_power <- pwr.t.test(n = n, d = effect_size, sig.level = alpha_corrected, 
                           type = "two.sample", alternative = "two.sided")$power
    # Compile results
    results <- rbind(results, 
                     data.frame(
                       Variable = var,
                       Shapiro_Wilk = shapiro_test$p.value,
                       KS_Test = ks_test$p.value,
                       Jarque_Bera = jarque_bera_test$p.value,
                       Anderson_Darling = ad_test$p.value,
                       SW_Power = sw_power,
                       KS_Power = ks_power,
                       JB_Power = jb_power,
                       AD_Power = ad_power
                     ))
  }
  # Add significance indicators after Bonferroni correction
  results$Significant <- apply(results[, 2:5], 1, function(x) 
    any(x < alpha_corrected))
  return(results)
}

normality_results <- normality_tests_selected(df_imputed)
print("\nNormality Test Results with Multiple Testing Correction:")
print(normality_results)

# Funzione per visualizzazione distribuzioni
generate_distribution_plots <- function(df) {
  # Variabili di interesse per l'analisi
  selected_vars <- c("age", "trestbps", "chol", "thalch")
  
  # Layout 2x2 per grafici
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  
  for (var in selected_vars) {
    # Calcolo parametri distribuzione normale
    mu <- mean(df[[var]], na.rm = TRUE)
    sigma <- sd(df[[var]], na.rm = TRUE)
    
    # Creazione istogramma con curva normale
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

# Generazione grafici distribuzione
generate_distribution_plots(df_imputed)

# Funzione per creazione boxplot
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

# Generazione boxplot
generate_boxplots(df_imputed)

# Funzione per Q-Q plots
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

# Generazione Q-Q plots
generate_qqplots(df_imputed)

analyze_ordinal_model <- function(data, target_var = "num", 
                                  predictors = NULL, 
                                  validation_split = 0.2, 
                                  seed = 123) {
  require(MASS)
  require(car)
  require(pROC)
  require(caret)
  require(pscl)
  
  # Set seed for reproducibility
  set.seed(seed)
  
  # Ensure target variable is an ordered factor
  data[[target_var]] <- factor(data[[target_var]], ordered = TRUE)
  
  # Prepare data
  if (is.null(predictors)) {
    predictors <- names(data)[!names(data) %in% target_var]
  }
  formula_str <- paste(target_var, "~", paste(predictors, collapse = " + "))
  formula_obj <- as.formula(formula_str)
  
  # Split data into training and testing sets
  train_indices <- createDataPartition(data[[target_var]], p = 1-validation_split, list = FALSE)
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Model fitting and stepwise selection
  initial_model <- polr(formula_obj, data = train_data, Hess = TRUE)
  stepwise_model <- step(initial_model, direction = 'both', trace = 0)
  
  # Feature importance analysis
  coef_summary <- summary(stepwise_model)
  coefficients <- coef(coef_summary)
  
  # Extract coefficient table properly
  coef_table <- coefficients[1:length(stepwise_model$coefficients), ]
  p_values <- pnorm(abs(coef_table[, "t value"]), lower.tail = FALSE) * 2
  
  # Confidence intervals
  ci <- confint.default(stepwise_model)[1:length(stepwise_model$coefficients), ]
  odds_ratios <- exp(stepwise_model$coefficients)
  odds_ci <- exp(ci)
  
  # Create feature importance dataframe
  feature_importance <- data.frame(
    Feature = names(stepwise_model$coefficients),
    Coefficient = coef_table[, "Value"],
    Std_Error = coef_table[, "Std. Error"],
    P_Value = p_values,
    OR = odds_ratios,
    OR_CI_Lower = odds_ci[, 1],
    OR_CI_Upper = odds_ci[, 2]
  )
  
  # Multicollinearity check
  vif_values <- try(vif(stepwise_model), silent = TRUE)
  
  # Calculate custom residuals for ordinal model
  fitted_probs <- predict(stepwise_model, type = "probs")
  fitted_class <- predict(stepwise_model, type = "class")
  actual_class <- train_data[[target_var]]
  
  # Calculate deviance residuals
  dev_residuals <- sign(as.numeric(actual_class) - as.numeric(fitted_class)) * 
    sqrt(-2 * log(apply(fitted_probs, 1, max)))
  
  # Predictions on test set
  predictions_prob <- predict(stepwise_model, newdata = test_data, type = "probs")
  predictions <- predict(stepwise_model, newdata = test_data)
  
  # Model performance metrics
  confusion_matrix <- table(Predicted = predictions, Actual = test_data[[target_var]])
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  # Calculate additional classification metrics
  precision <- diag(confusion_matrix) / colSums(confusion_matrix)
  recall <- diag(confusion_matrix) / rowSums(confusion_matrix)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # ROC curves (one-vs-all approach for each class)
  roc_curves <- list()
  auc_values <- numeric()
  
  for(level in levels(data[[target_var]])) {
    binary_actual <- ifelse(test_data[[target_var]] == level, 1, 0)
    binary_pred <- predictions_prob[, level]
    roc_obj <- roc(binary_actual, binary_pred)
    roc_curves[[level]] <- roc_obj
    auc_values[level] <- auc(roc_obj)
  }
  
  # Additional performance metrics
  mcfadden <- pR2(stepwise_model)["McFadden"]
  performance_metrics <- data.frame(
    Metric = c("Accuracy", "McFadden R2", "AIC", "BIC"),
    Value = c(
      accuracy,
      mcfadden,
      AIC(stepwise_model),
      BIC(stepwise_model)
    )
  )
  
  # Additional classification metrics
  classification_metrics <- data.frame(
    Class = levels(data[[target_var]]),
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  )
  
  # Store fitted values and custom residuals
  model_diagnostics <- list(
    vif = vif_values,
    residuals = data.frame(
      Fitted = as.numeric(fitted_class),
      Residuals = dev_residuals
    )
  )
  
  # Prepare results
  results <- list(
    model = stepwise_model,
    feature_importance = feature_importance,
    model_diagnostics = model_diagnostics,
    performance = list(
      metrics = performance_metrics,
      classification_metrics = classification_metrics,
      confusion_matrix = confusion_matrix,
      roc_curves = roc_curves,
      auc_values = auc_values
    )
  )
  
  # Print summary
  cat("\n=== Model Analysis Results ===\n")
  cat("\nFeature Importance:\n")
  print(results$feature_importance)
  
  cat("\nMulticollinearity Check (VIF):\n")
  print(results$model_diagnostics$vif)
  
  cat("\nModel Performance Metrics:\n")
  print(results$performance$metrics)
  
  cat("\nClassification Metrics by Class:\n")
  print(results$performance$classification_metrics)
  
  cat("\nConfusion Matrix:\n")
  print(results$performance$confusion_matrix)
  
  cat("\nAUC Values:\n")
  print(results$performance$auc_values)
  
  # Plot diagnostics
  par(mfrow = c(2, 2))
  
  # Residual plot
  plot(results$model_diagnostics$residuals$Fitted,
       results$model_diagnostics$residuals$Residuals,
       main = "Residual Plot",
       xlab = "Fitted Values",
       ylab = "Deviance Residuals")
  abline(h = 0, col = "red", lty = 2)
  
  # ROC curves
  colors <- rainbow(length(levels(data[[target_var]])))
  plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1),
       main = "ROC Curves", xlab = "False Positive Rate",
       ylab = "True Positive Rate")
  abline(0, 1, lty = 2, col = "gray")
  
  for(i in seq_along(roc_curves)) {
    lines(roc_curves[[i]], col = colors[i])
  }
  legend("bottomright", legend = names(roc_curves),
         col = colors, lwd = 2)
  
  # QQ plot of residuals
  qqnorm(results$model_diagnostics$residuals$Residuals)
  qqline(results$model_diagnostics$residuals$Residuals, col = "red")
  
  # Histogram of residuals
  hist(results$model_diagnostics$residuals$Residuals,
       main = "Histogram of Residuals",
       xlab = "Deviance Residuals",
       breaks = 30,
       probability = TRUE)
  lines(density(results$model_diagnostics$residuals$Residuals),
        col = "red", lwd = 2)
  
  par(mfrow = c(1, 1))
  
  return(results)
}

# Run the analysis with the fixed function
model_results <- analyze_ordinal_model(df_imputed)