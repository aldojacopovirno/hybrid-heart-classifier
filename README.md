1. Analisi Esplorativa dei Dati (EDA) dei Parametri Cinematici

	- Descrizione: 
	Conduci un’analisi statistica descrittiva delle variabili principali (MR, Rsq, HT, MET) per caratterizzare la distribuzione degli eventi. Esplora la correlazione tra queste variabili e identifica eventuali pattern o anomalie.

	- Approcci Statistici:
	Visualizzazione tramite istogrammi, scatter plot e heatmap.
	Correlazione lineare e non lineare tra le variabili.
	Test statistici per verificare la normalità delle distribuzioni
	
	- Obiettivo: 
	Fornire una descrizione dettagliata delle caratteristiche del dataset, identificando possibili segni di eventi anomali o significativi in un contesto di fisica delle particelle.

2. Modellazione di Regressione per Predire MR e Rsq

	- Descrizione: 
	Costruisci modelli di regressione per prevedere i valori delle variabili MR e Rsq basandoti su altre variabili cinematiche come HT, MET, nJets, e nBJets. Analizza quali variabili hanno il maggiore impatto su MR e Rsq.
		
	- Approcci Statistici:
	Regressione lineare multipla, Ridge, Lasso, e metodi di selezione delle variabili.
	Valutazione del modello attraverso il coefficiente di determinazione R^2, errore quadratico medio (MSE), e cross-validation.
	Feature importance per identificare le variabili più rilevanti.
	
	- Obiettivo:
	Presentare un modello statistico che descrive la relazione tra le caratteristiche degli eventi e le variabili MR e Rsq, fornendo intuizioni sui fattori determinanti per queste misure chiave.

3. Rilevamento di Anomalie Basato su MET e HT

	- Descrizione: 
	Implementa metodi di anomaly detection per identificare eventi con valori estremi di MET e HT, che potrebbero rappresentare segnali di nuove particelle o fisica oltre il Modello Standard.
	
	- Approcci Statistici:
	Algoritmi di rilevamento delle anomalie come Isolation Forest, One-Class SVM, e metodi di densità come Local Outlier Factor (LOF).
	Analisi delle anomalie identificate e loro confronto con eventi “normali”.
	
	- Obiettivo:
	Mostrare come tecniche avanzate di rilevamento delle anomalie possano essere applicate a eventi di fisica delle particelle per scoprire segnali rari o nuovi fenomeni.

5. Confronto Statistico tra Eventi ad Alto e Basso MET

	- Descrizione: 
	Esegui un confronto statistico tra eventi con alto MET e basso MET per vedere se ci sono differenze significative nelle altre variabili cinematiche (MR, Rsq, HT, nJets).
	
	- Approcci Statistici:
	Test statistici come il t-test per campioni indipendenti o il test di Mann-Whitney.
	Analisi delle differenze tra distribuzioni utilizzando test di Kolmogorov-Smirnov.
	Visualizzazione delle differenze tramite boxplot e violin plot.
	
	- Obiettivo:
	 Evidenziare differenze significative tra eventi ad alta e bassa energia trasversa mancante, suggerendo potenziali segnali di particelle invisibili o nuove particelle.

6. Test d’Ipotesi per la Ricerca di Nuove Particelle Supersimmetriche

	- Descrizione: 
	Testa l’ipotesi che alcuni eventi con valori estremi di MR e Rsq potrebbero essere dovuti a particelle supersimmetriche. Confronta la distribuzione osservata con quella attesa dal Modello Standard.
	
	- Approcci Statistici:
	Test di ipotesi, come il chi-quadrato o il likelihood ratio test.
	Simulazioni Monte Carlo per generare dati secondo le aspettative del Modello Standard e confrontarli con i dati osservati.
	
	- Obiettivo:
	Presentare un’analisi rigorosa delle variabili razor per verificare la possibilità di scoprire nuove particelle, confrontando dati reali con modelli teorici.
