# Analisi Statistica delle Malattie Cardiache

Questo progetto implementa un'analisi statistica completa di dati sulle malattie cardiache utilizzando R. Il codice fornisce un framework completo per l'analisi esplorativa dei dati (EDA), test di normalità, visualizzazione delle distribuzioni e modellazione predittiva attraverso regressione ordinale.

## Caratteristiche Principali

- Preprocessing completo dei dati
- Analisi statistica descrittiva
- Test di normalità multipli con correzione di Bonferroni
- Visualizzazioni statistiche (istogrammi, boxplot, Q-Q plot)
- Regressione ordinale per la predizione della severità della malattia
- Analisi ROC con curve per ogni classe
- Gestione automatica dei valori mancanti tramite KNN imputation

## Prerequisiti

Le seguenti librerie R sono necessarie:

```R
- readr      # Lettura file CSV
- labstatR   # Funzioni statistiche base
- tseries    # Analisi serie temporali
- moments    # Calcolo asimmetria e curtosi
- VIM        # Gestione dati mancanti
- gridExtra  # Layout grafici
- lmtest     # Test Jarque-Bera
- nortest    # Test di normalità
- MASS       # Regressione logistica ordinale
- car        # Diagnostica modelli
- olsrr      # Diagnostica modelli
- pscl       # Pseudo R-squared
- pwr        # Analisi potenza statistica
- dplyr      # Manipolazione dati
- caret      # Machine Learning
- pROC       # Curve ROC
```

## Come Iniziare

1. Clone del repository:
```bash
git clone [URL_DEL_TUO_REPOSITORY]
```

2. Assicurarsi che tutte le dipendenze siano installate:
```R
required_packages <- c("readr", "labstatR", "tseries", "moments", "VIM", 
                      "gridExtra", "lmtest", "nortest", "MASS", "car", 
                      "olsrr", "pscl", "pwr", "dplyr", "caret", "pROC")

for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}
```

3. Eseguire l'analisi completa:
```R
source("your_script.R")
results <- run_complete_analysis("heart_disease_uci.csv")
```

## Pipeline di Analisi

### 1. Preprocessing dei Dati
- Codifica delle variabili categoriche
- Gestione dei valori mancanti tramite KNN imputation
- Preparazione dei dati per la regressione ordinale

### 2. Analisi Statistica Base
- Statistiche descrittive complete
- Intervalli di confidenza
- Test di ipotesi
- Analisi degli outlier
- Matrici di correlazione

### 3. Test di Normalità
- Test di Shapiro-Wilk
- Test di Kolmogorov-Smirnov
- Test di Jarque-Bera
- Test di Anderson-Darling
- Correzione di Bonferroni per test multipli
- Analisi della potenza statistica

### 4. Visualizzazioni
- Istogrammi con curve di densità normale
- Boxplot per identificazione outlier
- Q-Q plot per valutazione normalità

### 5. Modellazione Predittiva
- Regressione logistica ordinale
- Analisi VIF per multicollinearità
- Metriche di valutazione del modello:
  - AIC/BIC
  - R² di McFadden
  - Matrice di confusione
  - Accuratezza, Precisione, Recall, F1-Score

### 6. Analisi ROC
- Curve ROC per ogni classe
- Calcolo AUC per ogni classe
- Visualizzazione comparativa delle performance

## Output dell'Analisi

Il codice produce una serie di output strutturati:

```R
results <- list(
    Preprocessed_Data = df_prepared,          # Dataset preprocessato
    Statistical_Analysis = stats_results,      # Risultati analisi statistica
    Normality_Tests = normality_results,       # Risultati test normalità
    Model_Results = model_results,             # Risultati modello predittivo
    ROC_Results = roc_results                  # Risultati analisi ROC
)
```

## Struttura del Dataset

Il dataset deve contenere le seguenti variabili:
- age: età del paziente
- sex: sesso del paziente
- cp: tipo di dolore toracico
- trestbps: pressione sanguigna a riposo
- chol: colesterolo
- fbs: glicemia a digiuno
- restecg: risultati elettrocardiogramma
- thalch: frequenza cardiaca massima
- exang: angina indotta da esercizio
- oldpeak: depressione ST
- slope: pendenza del segmento ST
- ca: numero di vasi principali
- thal: talassemia
- num: diagnosi di malattia cardiaca (variabile target)

## Licenza

Distribuito sotto licenza MIT. Vedere `LICENSE` per maggiori informazioni.

## Contatti

Aldo Jacopo Virno - aldojacopo@gmail.com
Andrea Bucchignani - andreabucchignani@gmail.com

Project Link: [https://github.com/username/repository-name]
