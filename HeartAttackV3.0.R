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
library(dplyr)
library(caret)


datamatrix <- read_csv("heart_disease_uci.csv")

cat("Structure of the dataset:\n")
str(datamatrix)

cat("\nFirst few rows of the dataset:\n")
print(head(datamatrix))


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

statistical_analysis <- function(df) {
  numeric_columns <- sapply(df, is.numeric)
  df_numeric <- df[, numeric_columns]
  n <- colSums(!is.na(df_numeric))
  alpha <- 0.025
  t_value <- qt(1 - alpha, df = n - 1)
  
  basic_stats <- data.frame(
    Variabile = names(df_numeric),
    Minimo = apply(df_numeric, 2, min, na.rm = TRUE),
    Q1 = apply(df_numeric, 2, function(x) quantile(x, probs = 0.25, na.rm = TRUE)),
    Mediana = apply(df_numeric, 2, median, na.rm = TRUE),
    Media = colMeans(df_numeric, na.rm = TRUE),
    Q3 = apply(df_numeric, 2, function(x) quantile(x, probs = 0.75, na.rm = TRUE)),
    Massimo = apply(df_numeric, 2, max, na.rm = TRUE),
    Varianza = apply(df_numeric, 2, var, na.rm = TRUE),
    Deviazione_Standard = apply(df_numeric, 2, sd, na.rm = TRUE),
    Skewness = sapply(df_numeric, skewness, na.rm = TRUE),
    Kurtosi = sapply(df_numeric, kurtosis, na.rm = TRUE)
  )
  
  lower_ci <- colMeans(df_numeric, na.rm = TRUE) - t_value * (apply(df_numeric, 2, sd, na.rm = TRUE) / sqrt(n))
  upper_ci <- colMeans(df_numeric, na.rm = TRUE) + t_value * (apply(df_numeric, 2, sd, na.rm = TRUE) / sqrt(n))
  
  p_values <- sapply(1:ncol(df_numeric), function(i) {
    t.test(df_numeric[, i], mu = 0, alternative = "two.sided")$p.value
  })
  
  p_values_sci <- formatC(p_values, format = "e", digits = 2)
  
  ci_pvalue_table <- data.frame(
    Variable = names(df_numeric),
    Lower_Ci = lower_ci,
    Upper_Ci = upper_ci,
    P_value = p_values_sci
  )
  
  outlier_detection <- function(x) {
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    outliers <- sum(x < lower_bound | x > upper_bound, na.rm = TRUE)
    return(outliers)
  }
  
  outliers_count <- sapply(df_numeric, outlier_detection)
  
  outliers_table <- data.frame(
    Variabile = names(df_numeric),
    Numero_Outlier = outliers_count
  )
  
  correlation_matrix <- cor(df_numeric, use = "complete.obs", method = "pearson")
  
  formatted_correlation_matrix <- round(correlation_matrix, 2)
  
  output <- list(
    "Statistiche Descrittive" = basic_stats,
    "Intervalli di Confidenza e P-value" = ci_pvalue_table,
    "Analisi degli Outlier" = outliers_table,
    "Matrice di Correlazione" = formatted_correlation_matrix
  )
  
  return(output)
}

normality_tests_selected <- function(df, selected_vars = c("age", "trestbps", "chol", "thalch"), 
                                     alpha = 0.05) {
  require(nortest)
  require(pwr)
  df_selected <- df[, selected_vars]
  n_tests <- length(selected_vars) * 4
  alpha_corrected <- alpha / n_tests
  results <- data.frame(
    Variabile = character(),
    Shapiro_Wilk_pvalue = numeric(),
    KS_Test_pvalue = numeric(),
    Jarque_Bera_pvalue = numeric(),
    Anderson_Darling_pvalue = numeric(),
    SW_Power = numeric(),
    KS_Power = numeric(),
    JB_Power = numeric(),
    AD_Power = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (var in names(df_selected)) {
    n <- sum(!is.na(df_selected[[var]]))
    
    shapiro_test <- shapiro.test(df_selected[[var]])
    ks_test <- ks.test(df_selected[[var]], "pnorm", 
                       mean = mean(df_selected[[var]], na.rm = TRUE),
                       sd = sd(df_selected[[var]], na.rm = TRUE))
    jarque_bera_test <- jarque.bera.test(df_selected[[var]])
    ad_test <- ad.test(df_selected[[var]])
    
    effect_size <- 0.3  
    sw_power <- pwr.t.test(n = n, d = effect_size, sig.level = alpha_corrected, 
                           type = "two.sample", alternative = "two.sided")$power
    ks_power <- pwr.t.test(n = n, d = effect_size, sig.level = alpha_corrected, 
                           type = "two.sample", alternative = "two.sided")$power
    jb_power <- pwr.t.test(n = n, d = effect_size, sig.level = alpha_corrected, 
                           type = "two.sample", alternative = "two.sided")$power
    ad_power <- pwr.t.test(n = n, d = effect_size, sig.level = alpha_corrected, 
                           type = "two.sample", alternative = "two.sided")$power
    
    results <- rbind(results, 
                     data.frame(
                       Variabile = var,
                       Shapiro_Wilk_pvalue = formatC(shapiro_test$p.value, format = "e", digits = 2),
                       KS_Test_pvalue = formatC(ks_test$p.value, format = "e", digits = 2),
                       Jarque_Bera_pvalue = formatC(jarque_bera_test$p.value, format = "e", digits = 2),
                       Anderson_Darling_pvalue = formatC(ad_test$p.value, format = "e", digits = 2),
                       SW_Power = round(sw_power, 3),
                       KS_Power = round(ks_power, 3),
                       JB_Power = round(jb_power, 3),
                       AD_Power = round(ad_power, 3)
                     ))
  }
  
  results$Significativo <- apply(results[, 2:5], 1, function(x) 
    any(as.numeric(x) < alpha_corrected))
  
  output <- list(
    "Risultati Test di Normalità" = results[, 1:5],
    "Potenza dei Test" = results[, c("Variabile", "SW_Power", "KS_Power", "JB_Power", "AD_Power")],
    "Soglia di Significatività Corrette (Bonferroni)" = alpha_corrected
  )
  
  return(output)
}

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

# Funzione per costruire e analizzare un modello di regressione ordinale
ordinal_model <- function(df, response_var, predictors) {
  # Assicurati che la variabile di risposta sia un fattore ordinato
  df[[response_var]] <- factor(df[[response_var]], ordered = TRUE)
  
  # Controlla se i predittori sono presenti nel dataframe
  missing_predictors <- setdiff(predictors, names(df))
  if(length(missing_predictors) > 0) {
    stop(paste("Le seguenti variabili non sono presenti nel dataframe:", 
               paste(missing_predictors, collapse = ", ")))
  }
  
  # Costruzione della formula
  formula <- as.formula(paste(response_var, "~", paste(predictors, collapse = " + ")))
  
  # Fit del modello di regressione ordinale
  ordinal_model <- polr(formula, data = df, Hess = TRUE)
  
  # Analisi del modello
  model_summary <- summary(ordinal_model)
  
  # Diagnostica della multicollinearità
  vif_values <- vif(ordinal_model)
  
  # Criteri di informazione
  aic_value <- AIC(ordinal_model)
  bic_value <- BIC(ordinal_model)
  
  # Calcolo R² di McFadden
  mcfadden_r2 <- pR2(ordinal_model)
  
  # Calcolo delle probabilità previste
  predicted_probs <- predict(ordinal_model, type = "probs")
  
  # Calcolo della matrice di confusione e metriche di valutazione
  predicted_classes <- apply(predicted_probs, 1, which.max)
  confusion_matrix <- table(Actual = df[[response_var]], 
                            Predicted = factor(predicted_classes, levels = 1:nlevels(df[[response_var]])))
  
  # Calcolo delle metriche di valutazione
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  # Precisione, Recall e F1 Score per ogni classe
  precision <- diag(confusion_matrix) / rowSums(confusion_matrix)
  recall <- diag(confusion_matrix) / colSums(confusion_matrix)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Preparazione dei risultati
  evaluation_metrics <- data.frame(
    Class = levels(df[[response_var]]),
    Precision = round(precision, 4),
    Recall = round(recall, 4),
    F1_Score = round(f1_score, 4)
  )
  
  # Organizza i risultati in una lista
  results <- list(
    Model_Summary = model_summary,
    VIF_Values = vif_values,
    AIC = aic_value,
    BIC = bic_value,
    McFadden_R2 = mcfadden_r2,
    Confusion_Matrix = confusion_matrix,
    Accuracy = round(accuracy, 4),
    Evaluation_Metrics = evaluation_metrics
  )
  
  return(results)
}

prepare_data <- function(df) {
  # Convert response variable to ordered factor
  df$num <- factor(df$num, ordered = TRUE, 
                   levels = sort(unique(df$num)))
  return(df)
}

# Function to generate ROC curves for ordinal model
generate_roc_curves <- function(model, data, response_var) {
  require(pROC)
  require(ggplot2)
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
    # Calculate ROC curve
    roc_obj <- roc(binary_actual, pred_probs[,i])
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
  # Return AUC values
  return(list(
    ROC_Objects = roc_list,
    AUC_Values = auc_values
  ))
}

# Prepare the data
df_prepared <- prepare_data(df_imputed)

# Create the model
model_formula <- as.formula("num ~ age + sex + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal")
ordinal_model <- polr(model_formula, data = df_prepared, Hess = TRUE)

# Generate ROC curves and get results
roc_results <- generate_roc_curves(ordinal_model, df_prepared, "num")

# Print AUC values
cat("\nAUC Values for each class:\n")
print(data.frame(
  Class = 1:length(roc_results$AUC_Values),
  AUC = round(roc_results$AUC_Values, 3)
))

# Print model summary
print(summary(ordinal_model))

df_encoded <- encoder(datamatrix)
df_imputed <- handle_missing_values(df_encoded)

print(statistical_analysis(df_imputed))
print(normality_tests_selected(df_imputed))

results <- ordinal_model(df_imputed, response_var = "num", 
                                      predictors = c("age", "sex", "cp",
                                                     "trestbps","chol",
                                                     "fbs", "restecg",
                                                     "thalch", "exang",
                                                     "oldpeak", "slope",
                                                     "ca", "thal"))
View(results$Model_Summary)
print(results$VIF_Values)
print(results$AIC)
print(results$BIC)
print(results$McFadden_R2)
print(results$Confusion_Matrix)
print(results$Accuracy)
print(results$Evaluation_Metrics)

generate_distribution_plots(df_imputed)
generate_boxplots(df_imputed)
generate_qqplots(df_imputed)
generate_roc_curves(results, df_imputed, factor(df_imputed$num, ordered = TRUE))
