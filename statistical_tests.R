#' Perform Statistical Analysis
#' 
#' @description
#' Conducts comprehensive statistical analysis on numeric variables.
#' 
#' @param df A data frame containing numeric variables
#' @return A list containing:
#'   \item{Statistiche Descrittive}{Basic statistics including mean, median, etc.}
#'   \item{Intervalli di Confidenza e P-value}{Confidence intervals and t-test results}
#'   \item{Analisi degli Outlier}{Outlier counts using IQR method}
#'   \item{Matrice di Correlazione}{Pearson correlation matrix}
#' @details
#' Calculates:
#' - Basic descriptive statistics
#' - 95% confidence intervals
#' - Two-sided t-tests against mu=0
#' - Outlier detection using 1.5*IQR rule
#' - Correlation analysis
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

#' Perform Normality Tests on Selected Variables
#' 
#' @description
#' Conducts multiple normality tests on specified variables.
#' 
#' @param df A data frame containing the variables to test
#' @param selected_vars Vector of variable names to test, default: c("age", "trestbps", "chol", "thalch")
#' @param alpha Significance level, default: 0.05
#' @return A list containing:
#'   \item{Risultati Test di Normalità}{Results of multiple normality tests}
#'   \item{Potenza dei Test}{Power analysis results}
#'   \item{Soglia di Significatività}{Bonferroni-corrected significance level}
#' @details
#' Performs:
#' - Shapiro-Wilk test
#' - Kolmogorov-Smirnov test
#' - Jarque-Bera test
#' - Anderson-Darling test
normality_tests_selected <- function(df, selected_vars = c("age", "trestbps", "chol", "thalch"), 
                                     alpha = 0.05) {
  require(nortest)
  require(pwr)
  
  # Suppress warnings specifically for KS test
  options(warn = -1)  # Temporarily suppress all warnings
  
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
    
    # Add small random noise to break ties for KS test
    data_with_noise <- df_selected[[var]] + rnorm(length(df_selected[[var]]), 0, sd(df_selected[[var]], na.rm = TRUE) / 1000)
    
    shapiro_test <- shapiro.test(df_selected[[var]])
    ks_test <- ks.test(data_with_noise, "pnorm", 
                       mean = mean(data_with_noise, na.rm = TRUE),
                       sd = sd(data_with_noise, na.rm = TRUE))
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
  
  # Reset warning option
  options(warn = 0)
  
  results$Significativo <- apply(results[, 2:5], 1, function(x) 
    any(as.numeric(x) < alpha_corrected))
  
  output <- list(
    "Risultati Test di Normalità" = results[, 1:5],
    "Potenza dei Test" = results[, c("Variabile", "SW_Power", "KS_Power", "JB_Power", "AD_Power")],
    "Soglia di Significatività Corrette (Bonferroni)" = alpha_corrected
  )
  
  return(output)
}