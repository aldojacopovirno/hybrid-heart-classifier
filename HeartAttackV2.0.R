# Impostazione directory di lavoro 
# TODO: Considerare l'uso di here::here() per maggiore portabilità
setwd("~/Code Nest/PWEDA-CodeNest")

# Caricamento librerie necessarie
library(readr)      # Per lettura file CSV
library(labstatR)   # Per funzioni statistiche base
library(tseries)    # Per analisi serie temporali e test statistici (caricato due volte)
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
summary(datamatrix)

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
statistical_analysis <- function(df) {
  # Select only numeric columns
  numeric_columns <- sapply(df, is.numeric)
  df_numeric <- df[, numeric_columns]
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
  # Confidence intervals for means (95%)
  ci_means <- t(apply(df_numeric, 2, function(x) {
    n <- sum(!is.na(x))
    se <- sd(x, na.rm = TRUE) / sqrt(n)
    ci <- qt(0.975, df = n - 1) * se
    c(mean(x, na.rm = TRUE) - ci, mean(x, na.rm = TRUE) + ci)
  }))
  colnames(ci_means) <- c("CI_Lower", "CI_Upper")
  # Outlier detection using IQR method
  outliers <- lapply(df_numeric, function(x) {
    q1 <- quantile(x, 0.25, na.rm = TRUE)
    q3 <- quantile(x, 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    outliers <- sum(x < lower_bound | x > upper_bound, na.rm = TRUE)
    outlier_prop <- outliers / length(x[!is.na(x)])
    c(n_outliers = outliers, prop_outliers = outlier_prop)
  })
  outliers_df <- as.data.frame(do.call(rbind, outliers))
  # Correlation analysis
  cor_matrix <- cor(df_numeric, use = "pairwise.complete.obs")
  # Compile results
  results <- list(
    descriptive_stats = basic_stats,
    confidence_intervals = ci_means,
    outliers = outliers_df,
    correlation_matrix = cor_matrix
  )
  return(results)
}

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

# Example usage:
stats_results <- statistical_analysis(df_imputed)
print("Descriptive Statistics:")
print(stats_results$descriptive_stats)
print("\nConfidence Intervals:")
print(stats_results$confidence_intervals)
print("\nOutlier Analysis:")
print(stats_results$outliers)
print("\nCorrelation Matrix:")
print(stats_results$correlation_matrix)

normality_results <- normality_tests_selected(df_imputed)
print("\nNormality Test Results with Multiple Testing Correction:")
print(normality_results)

# Funzione per visualizzazione distribuzioni
# TODO: Considerare uso di ggplot2 per maggiore flessibilità
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
# TODO: Aggiungere visualizzazione violin plot
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

# Preparazione per regressione logistica ordinale
# Conversione variabile target in fattore ordinato
df_imputed$num <- factor(df_imputed$num, ordered = TRUE)

# Creazione modello di regressione logistica ordinale
# TODO: Aggiungere validazione incrociata
ordinal_model <- polr(num ~ ., data = df_imputed, Hess = TRUE)

# Analisi modello
summary(ordinal_model)

# Selezione stepwise del modello
step(ordinal_model, direction = 'both')

# Diagnostica multicollinearità
vif(ordinal_model)

# Criteri di informazione
AIC(ordinal_model)
BIC(ordinal_model)

# Calcolo R² di McFadden
mcfadden_r2 <- pR2(ordinal_model)
print(mcfadden_r2)

# Estrazione coefficienti e p-value
ctable <- coef(summary(ordinal_model))
ctable1 <- ordinal_model$coefficients

print(ctable1)

# Calcolo p-value
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
ctable <- cbind(ctable, "p value" = p)

# Calcolo intervalli di confidenza
ci_logit <- confint.default(ordinal_model)
ci_odds <- exp(confint.default(ordinal_model))