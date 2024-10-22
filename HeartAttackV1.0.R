# Set working directory
setwd("~/Code Nest/PWEDA-CodeNest")

# Load necessary libraries for data manipulation, statistical analysis, imputation, and visualization.
library(readr)      # For reading CSV files
library(labstatR)   # For statistical functions (e.g., skewness, kurtosis)
library(tseries)    # For additional time series and statistical analysis
library(moments)    # For calculating skewness and kurtosis
library(VIM)        # For handling missing data (KNN imputation)
library(gridExtra)  # For arranging multiple plots in a grid
library(tseries)    # For the Kolmogorov-Smirnov test
library(lmtest)     # For the Jarque-Bera test
library(nortest)
library(MASS)
library(car)
library(olsrr)
library(pscl)

# Load the dataset from a CSV file
datamatrix <- read_csv("heart_disease_uci.csv")

# Print the structure of the dataset to show column names, types, and overall structure
cat("Structure of the dataset:\n")
str(datamatrix)

# Print the first few rows of the dataset to get a quick overview
cat("\nFirst few rows of the dataset:\n")
print(head(datamatrix))

# Funzione per la trasformazione delle variabili categoriche in numeriche
encoder <- function(df) {
  # Creiamo una copia del dataset per non modificare l'originale
  transformed_data <- df
  # Conversione del sesso
  transformed_data$sex <- ifelse(transformed_data$sex == "Male", 1, 0)
  # Conversione del tipo di dolore toracico (cp)
  transformed_data$cp <- factor(transformed_data$cp, 
                                levels = c("typical angina", "atypical angina", "non-anginal", "asymptomatic"))
  transformed_data$cp <- as.numeric(transformed_data$cp) - 1
  # Conversione del risultato ECG a riposo (restecg)
  transformed_data$restecg <- factor(transformed_data$restecg, 
                                     levels = c("normal", "st-t wave abnormality", "lv hypertrophy"))
  transformed_data$restecg <- as.numeric(transformed_data$restecg) - 1
  # Conversione del valore fbs
  transformed_data$fbs <- as.numeric(transformed_data$fbs)
  # Conversione dell'angina indotta da esercizio (exang)
  transformed_data$exang <- as.numeric(transformed_data$exang)
  # Conversione della pendenza del segmento ST (slope)
  transformed_data$slope <- factor(transformed_data$slope, 
                                   levels = c("upsloping", "flat", "downsloping"))
  transformed_data$slope <- as.numeric(transformed_data$slope) - 1
  # Conversione del difetto di perfusione (thal)
  transformed_data$thal <- factor(transformed_data$thal, 
                                  levels = c("normal", "fixed defect", "reversable defect"))
  transformed_data$thal <- as.numeric(transformed_data$thal) - 1
  # Gestione dei valori mancanti in 'ca'
  transformed_data$ca <- as.numeric(as.character(transformed_data$ca))
  transformed_data$ca[is.na(transformed_data$ca)] <- median(transformed_data$ca, na.rm = TRUE)
  return(transformed_data)
}

# Apply the encoder function
df_encoded <- encoder(datamatrix)

# Funzione per controllare e gestire i valori mancanti utilizzando KNN Imputer
handle_missing_values <- function(df) {
  # Controlla la presenza di valori mancanti
  missing_values <- sapply(df, function(x) sum(is.na(x)))
  
  # Stampa i risultati
  cat("Valori mancanti per colonna:\n")
  print(missing_values[missing_values > 0])
  
  # Se ci sono valori mancanti, esegui KNN Imputer
  if (sum(missing_values) > 0) {
    library(VIM)  # Assicurati che la libreria VIM sia caricata
    
    # Esegui KNN Imputer
    imputed_data <- kNN(df, k = 5, imp_var = FALSE)  # Non serve 'train'
    
    return(imputed_data)
  } else {
    cat("Nessun valore mancante trovato nel dataset.\n")
    return(df)
  }
}

# Apply the function to handle missing values
df_imputed <- handle_missing_values(df_encoded)

# Funzione per analisi statistica delle variabili quantitative
statistical_analysis <- function(df) {
  # Seleziona solo colonne numeriche
  numeric_columns <- sapply(df, is.numeric)
  
  # Estrae solo le colonne numeriche
  df_numeric <- df[, numeric_columns]
  
  # Calcola le statistiche di base
  means <- colMeans(df_numeric, na.rm = TRUE)
  medians <- apply(df_numeric, 2, median, na.rm = TRUE)
  sds <- apply(df_numeric, 2, sd, na.rm = TRUE)
  skewness_values <- sapply(df_numeric, skewness, na.rm = TRUE)
  kurtosis_values <- sapply(df_numeric, kurtosis, na.rm = TRUE)
  
  # Crea un dataframe per i risultati
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

# Esegui l'analisi statistica e stampa i risultati

analysis_results <- statistical_analysis(df_imputed)
cat("\nRisultati dell'analisi statistica:\n")
print(analysis_results)

# Funzione per generare histogrammi delle distribuzioni con sovrapposizione della distribuzione normale
generate_distribution_plots <- function(df) {
  # Seleziona le colonne specifiche
  selected_vars <- c("age", "trestbps", "chol", "thalch")
  
  # Imposta la grafica per visualizzare i grafici in un layout 2x2
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  
  for (var in selected_vars) {
    # Calcola i parametri della distribuzione normale
    mu <- mean(df[[var]], na.rm = TRUE)     # Media
    sigma <- sd(df[[var]], na.rm = TRUE)    # Deviazione standard
    
    # Istogramma della variabile
    hist(df[[var]], 
         main = paste("Histogram of", var), 
         xlab = var, 
         ylab = "Frequency", 
         probability = TRUE,          # Normalizza l'asse y
         col = "lightblue", 
         border = "black")
    
    # Aggiungi la curva della distribuzione normale
    curve(dnorm(x, mean = mu, sd = sigma), 
          add = TRUE, 
          col = "red", 
          lwd = 2)
  }
  
  # Ripristina il layout grafico a uno solo
  par(mfrow = c(1, 1))
}

# Genera i grafici delle distribuzioni per le variabili selezionate
generate_distribution_plots(df_imputed)

# Funzione per generare boxplot delle variabili selezionate
generate_boxplots <- function(df) {
  # Seleziona le colonne specifiche
  selected_vars <- c("age", "trestbps", "chol", "thalch")
  
  # Imposta la grafica per visualizzare i boxplot in un layout 2x2
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  
  for (var in selected_vars) {
    # Boxplot della variabile
    boxplot(df[[var]], 
            main = paste("Boxplot of", var), 
            ylab = var, 
            col = "lightblue", 
            border = "black", 
            notch = TRUE)
  }
  
  # Ripristina il layout grafico a uno solo
  par(mfrow = c(1, 1))
}

# Genera i boxplot per le variabili selezionate
generate_boxplots(df_imputed)

# Funzione per eseguire i test di normalità sulle variabili selezionate
normality_tests_selected <- function(df) {
  # Seleziona solo le colonne specifiche
  selected_vars <- c("age", "trestbps", "chol", "thalch")
  df_selected <- df[, selected_vars]
  
  # Inizializza un dataframe per i risultati
  normality_results <- data.frame(Variable = character(),
                                  Shapiro_Wilk = numeric(),
                                  KS_Test = numeric(),
                                  Jarque_Bera = numeric(),
                                  Anderson_Darling = numeric(),
                                  stringsAsFactors = FALSE)
  
  # Esegui i test per ogni variabile selezionata
  for (var in names(df_selected)) {
    # Shapiro-Wilk test
    shapiro_test <- shapiro.test(df_selected[[var]])$p.value
    
    # Kolmogorov-Smirnov test
    ks_test <- ks.test(df_selected[[var]], "pnorm", 
                       mean = mean(df_selected[[var]], na.rm = TRUE), 
                       sd = sd(df_selected[[var]], na.rm = TRUE))$p.value
    
    # Jarque-Bera test
    jarque_bera_test <- jarque.bera.test(df_selected[[var]])$p.value
    
    # Anderson-Darling test
    ad_test <- ad.test(df_selected[[var]])$p.value
    
    # Aggiungi i risultati al dataframe
    normality_results <- rbind(normality_results, 
                               data.frame(Variable = var,
                                          Shapiro_Wilk = shapiro_test,
                                          KS_Test = ks_test,
                                          Jarque_Bera = jarque_bera_test,
                                          Anderson_Darling = ad_test))
  }
  
  return(normality_results)
}

# Esegui i test di normalità sulle variabili selezionate e stampa i risultati
normality_results_selected <- normality_tests_selected(df_imputed)
cat("\nRisultati dei test di normalità per le variabili selezionate:\n")
print(normality_results_selected)

# Funzione per generare Q-Q plots per le variabili selezionate
generate_qqplots <- function(df) {
  # Seleziona le colonne specifiche
  selected_vars <- c("age", "trestbps", "chol", "thalch")
  
  # Imposta la grafica per visualizzare i Q-Q plots in un layout 2x2
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  
  for (var in selected_vars) {
    # Q-Q plot
    qqnorm(df[[var]], main = paste("Q-Q Plot for", var), 
           xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
    qqline(df[[var]], col = "red", lwd = 2)  # Aggiungi la linea di riferimento
  }
  
  # Ripristina il layout grafico a uno solo
  par(mfrow = c(1, 1))
}

# Genera i Q-Q plots per le variabili selezionate
generate_qqplots(df_imputed)


# Modello di regressione logistica ordinale
# Assicurati che la variabile Num sia un fattore ordinato
df_imputed$num <- factor(df_imputed$num, ordered = TRUE)

# Fit the ordinal logistic regression model
ordinal_model <- polr(num ~ ., data = df_imputed, Hess = TRUE)

# Summary of the model
summary(ordinal_model)

step(ordinal_model, direction = 'both')

vif(fit)

# McFadden's Pseudo R-squared
mcfadden_r2 <- pR2(fit)
print(mcfadden_r2)