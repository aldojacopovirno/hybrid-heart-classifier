### Roadmap EDA per Dataset sulle Malattie Cardiache (Sviluppo in R)

#### 1. **Setup Iniziale del Progetto**
- **Caricamento delle Librerie**: Installare e caricare le librerie principali come `tidyverse`, `ggplot2`, `dplyr`, `readr`, e `summarytools` per la manipolazione dei dati e la visualizzazione.
- **Caricamento del Dataset**: Importare il dataset in R utilizzando funzioni come `read.csv()` o `read.table()` e visualizzare le prime righe con `head()` per una panoramica iniziale.

#### 2. **Esplorazione Iniziale dei Dati**
- **Ispezionare la Struttura del Dataset**: Utilizzare `str()` per analizzare la struttura dei dati, controllando i tipi di variabili e l'eventuale presenza di NA.
- **Statistica Descrittiva**: Ottenere statistiche descrittive di base con funzioni come `summary()` e `describe()` per capire la distribuzione delle variabili numeriche.
- **Esplorazione delle Variabili Categoriali**: Verificare i livelli delle variabili categoriali con `table()` per controllare la loro frequenza e la distribuzione.

#### 3. **Pulizia dei Dati**
- **Trattamento dei Valori Mancanti**: Identificare i valori mancanti usando `is.na()` e decidere una strategia, come l’imputazione (media/mediana per variabili numeriche o moda per categoriali) o l’eliminazione dei campioni/variabili con troppi valori mancanti.
- **Gestione degli Outlier**: Identificare outlier tramite box plot o funzioni come `boxplot()` e valutare se rimuoverli o trasformarli.
- **Verifica delle Variabili Categoriali**: Assicurarsi che tutte le variabili categoriali siano nel formato corretto, convertendo variabili numeriche categoriali in `factor()` dove necessario.

#### 4. **Analisi Univariata**
- **Distribuzione delle Variabili Numeriche**: Utilizzare istogrammi e densità per visualizzare le distribuzioni delle variabili numeriche, tramite `ggplot2` o `hist()`.
- **Analisi delle Variabili Categoriali**: Creare grafici a barre per le variabili categoriali per capire la distribuzione dei dati con `ggplot2` o `barplot()`.

#### 5. **Analisi Bivariata**
- **Relazione tra Variabili Numeriche e Target**: Utilizzare scatter plot per esaminare la relazione tra variabili numeriche e la variabile target (ad es. `num`), accompagnati dal calcolo della correlazione (coefficiente di Pearson).
- **Relazione tra Variabili Categoriali e Target**: Creare grafici a barre o tabelle di contingenza per esaminare la relazione tra le variabili categoriali e la variabile target.
- **Heatmap di Correlazione**: Creare una matrice di correlazione per variabili numeriche e visualizzarla come heatmap per esplorare le correlazioni tra variabili e identificare multicollinearità.

#### 6. **Visualizzazione dei Dati**
- **Grafici a Dispersione**: Usare scatter plot per visualizzare relazioni tra coppie di variabili numeriche.
- **Box Plot**: Utilizzare box plot per comparare la distribuzione delle variabili numeriche in relazione alla variabile target.
- **Heatmap**: Visualizzare la matrice di correlazione con una heatmap per evidenziare relazioni significative tra le variabili.

#### 7. **Feature Engineering**
- **Codifica delle Variabili Categoriali**: Se necessario, convertire le variabili categoriali in numeriche utilizzando dummy variables con `model.matrix()` o `mutate()` di `dplyr`.
- **Creazione di Nuove Variabili**: Valutare la possibilità di creare nuove variabili o feature derivate dalle variabili esistenti, come rapporti tra variabili o trasformazioni logaritmiche, se opportuno.

#### 8. **Analisi Multivariata**
- **Relazione tra Più Variabili e il Target**: Esplorare come variabili multiple interagiscono tra loro e con il target attraverso l’uso di tecniche di regressione o analisi delle componenti principali (PCA), solo come strumento esplorativo.

---

### Stima della Probabilità di Sviluppare una Malattia Cardiaca

#### 9. **Modello di Regressione Logistica**
Poiché il target (`num`) sembra indicare un esito binario o categoriale (presenza o assenza di malattia cardiaca), una **regressione logistica** può essere utilizzata per stimare la probabilità di sviluppare la malattia.

1. **Trasformazione della Variabile Target**: Se `num` è un valore continuo o ha più categorie, ridurre le categorie a una forma binaria per la stima di probabilità.
   
2. **Creazione del Modello di Regressione Logistica**:
   - Usare `glm()` per eseguire una regressione logistica, specificando il target come variabile dipendente e le altre variabili del dataset come predittori.
   
3. **Valutazione del Modello**:
   - **Coefficiente di Regressione**: Esaminare i coefficienti del modello per interpretare l’impatto di ogni variabile predittiva sulla probabilità di sviluppare la malattia.
   - **Statistica Pseudo-R²**: Valutare la bontà del modello con misure come AIC o pseudo-R².
   - **Curva ROC**: Tracciare la curva ROC per valutare la capacità predittiva del modello, calcolando l’AUC (Area Under Curve) per misurare la performance.

4. **Previsione delle Probabilità**:
   - Usare il modello per stimare la probabilità di malattia per ciascun paziente e interpretare i risultati.
   
5. **Validazione del Modello**:
   - Dividere il dataset in training e test set per validare le performance del modello e prevenire overfitting.
   - Utilizzare la metrica **accuratezza** e l’analisi della matrice di confusione per valutare le previsioni.

#### 10. **Conclusioni Finali**
- **Sintesi dei Risultati**: Riassumere le intuizioni principali emerse dall'EDA e dall’analisi delle variabili più significative.
- **Implicazioni per la Stima del Rischio**: Presentare i risultati della stima di probabilità, sottolineando quali fattori aumentano maggiormente il rischio di malattia cardiaca.

---

### Tabella delle Variabili

| Variabile  | Descrizione                                                                                  |
|------------|----------------------------------------------------------------------------------------------|
| `id`       | ID univoco del paziente                                                                      |
| `age`      | Età del paziente in anni                                                                     |
| `origin`   | Provenienza del campione di studio                                                           |
| `sex`      | Sesso del paziente (Maschio/Femmina)                                                         |
| `cp`       | Tipo di dolore toracico (angina tipica, angina atipica, non anginale, asintomatico)           |
| `trestbps` | Pressione sanguigna a riposo (mm Hg all’ammissione in ospedale)                              |
| `chol`     | Colesterolo nel siero (mg/dl)                                                                |
| `fbs`      | Glicemia a digiuno (se > 120 mg/dl)                                                          |
| `restecg`  | Risultati dell’elettrocardiogramma a riposo (normale, anomalia ST-T, ipertrofia ventricolare) |
| `thalach`  | Frequenza cardiaca massima raggiunta                                                         |
| `exang`    | Angina indotta da esercizio (True/False)                                                     |
| `oldpeak`  | Depressione ST indotta dall’esercizio rispetto al riposo                                      |
| `slope`    | Pendenza del segmento ST durante esercizio                                                   |
| `ca`       | Numero di vasi principali (0-3) colorati con fluoroscopia                                    |
| `thal`     | Esito del test del talio (normale, difetto fisso, difetto reversibile)                       |
| `num`      | Variabile target che predice la malattia cardiaca                                             |