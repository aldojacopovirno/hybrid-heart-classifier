# Impostazione directory di lavoro 
# TODO: Considerare l'uso di here::here() per maggiore portabilità
setwd("~/Code Nest/PWEDA-CodeNest")

# Caricamento librerie necessarie
# Nota: alcune librerie potrebbero essere ridondanti - valutare l'eliminazione di duplicati
library(readr)      # Per lettura file CSV
library(labstatR)   # Per funzioni statistiche base
library(tseries)    # Per analisi serie temporali e test statistici (caricato due volte)
library(moments)    # Per calcolo asimmetria e curtosi
library(VIM)        # Per gestione dati mancanti (KNN imputation)
library(gridExtra)  # Per layout grafici multipli
library(tseries)    # Duplicato - rimuovere
library(lmtest)     # Per test Jarque-Bera
library(nortest)    # Per test di normalità aggiuntivi
library(MASS)       # Per regressione logistica ordinale
library(car)        # Per diagnostica modelli (VIF)
library(olsrr)      # Per diagnostica modelli
library(pscl)       # Per pseudo R-squared

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

# Funzione per analisi statistica descrittiva
# TODO: Aggiungere intervalli di confidenza
statistical_analysis <- function(df) {
  # Selezione colonne numeriche
  numeric_columns <- sapply(df, is.numeric)
  df_numeric <- df[, numeric_columns]
  
  # Calcolo statistiche descrittive
  means <- colMeans(df_numeric, na.rm = TRUE)
  medians <- apply(df_numeric, 2, median, na.rm = TRUE)
  sds <- apply(df_numeric, 2, sd, na.rm = TRUE)
  skewness_values <- sapply(df_numeric, skewness, na.rm = TRUE)
  kurtosis_values <- sapply(df_numeric, kurtosis, na.rm = TRUE)
  
  # Creazione dataframe risultati
  analysis_results <- data.frame(
    Variable = names(df_numeric),
    Mean = means,
    Median = medians,
    Standard_Deviation = sds,
    Skewness = skewness_values,
    Kurtosis = kurtosis_values
  )
  
  return(analysis_results)
}

# Esecuzione analisi statistica
analysis_results <- statistical_analysis(df_imputed)
cat("\nRisultati dell'analisi statistica:\n")
print(analysis_results)

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

# Funzione per test di normalità
# TODO: Aggiungere correzione per test multipli
normality_tests_selected <- function(df) {
  selected_vars <- c("age", "trestbps", "chol", "thalch")
  df_selected <- df[, selected_vars]
  
  normality_results <- data.frame(
    Variable = character(),
    Shapiro_Wilk = numeric(),
    KS_Test = numeric(),
    Jarque_Bera = numeric(),
    Anderson_Darling = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (var in names(df_selected)) {
    # Esecuzione test di normalità
    shapiro_test <- shapiro.test(df_selected[[var]])$p.value
    
    ks_test <- ks.test(df_selected[[var]], "pnorm", 
                       mean = mean(df_selected[[var]], na.rm = TRUE), 
                       sd = sd(df_selected[[var]], na.rm = TRUE))$p.value
    
    jarque_bera_test <- jarque.bera.test(df_selected[[var]])$p.value
    
    ad_test <- ad.test(df_selected[[var]])$p.value
    
    # Aggregazione risultati
    normality_results <- rbind(normality_results, 
                               data.frame(Variable = var,
                                          Shapiro_Wilk = shapiro_test,
                                          KS_Test = ks_test,
                                          Jarque_Bera = jarque_bera_test,
                                          Anderson_Darling = ad_test))
  }
  
  return(normality_results)
}

# Esecuzione test normalità
normality_results_selected <- normality_tests_selected(df_imputed)
cat("\nRisultati dei test di normalità per le variabili selezionate:\n")
print(normality_results_selected)

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