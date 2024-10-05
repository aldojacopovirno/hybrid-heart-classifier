**Goals**

1. **Analisi Esplorativa dei Dati (EDA) dei Parametri Cinematici**

	- Descrizione: 
	Conduci un’analisi statistica descrittiva delle variabili principali (MR, Rsq, HT, MET) per caratterizzare la distribuzione degli eventi. Esplora la correlazione tra queste variabili e identifica eventuali pattern o anomalie.

	- Approcci Statistici:
	Visualizzazione tramite istogrammi, scatter plot e heatmap.
	Correlazione lineare e non lineare tra le variabili.
	Test statistici per verificare la normalità delle distribuzioni
	
	- Obiettivo: 
	Fornire una descrizione dettagliata delle caratteristiche del dataset, identificando possibili segni di eventi anomali o significativi in un contesto di fisica delle particelle.

2. **Modellazione di Regressione per Predire MR e Rsq**

	- Descrizione: 
	Costruisci modelli di regressione per prevedere i valori delle variabili MR e Rsq basandoti su altre variabili cinematiche come HT, MET, nJets, e nBJets. Analizza quali variabili hanno il maggiore impatto su MR e Rsq.
		
	- Approcci Statistici:
	Regressione lineare multipla, Ridge, Lasso, e metodi di selezione delle variabili.
	Valutazione del modello attraverso il coefficiente di determinazione R^2, errore quadratico medio (MSE), e cross-validation.
	Feature importance per identificare le variabili più rilevanti.
	
	- Obiettivo:
	Presentare un modello statistico che descrive la relazione tra le caratteristiche degli eventi e le variabili MR e Rsq, fornendo intuizioni sui fattori determinanti per queste misure chiave.

3. **Rilevamento di Anomalie Basato su MET e HT**

	- Descrizione: 
	Implementa metodi di anomaly detection per identificare eventi con valori estremi di MET e HT, che potrebbero rappresentare segnali di nuove particelle o fisica oltre il Modello Standard.
	
	- Approcci Statistici:
	Algoritmi di rilevamento delle anomalie come Isolation Forest, One-Class SVM, e metodi di densità come Local Outlier Factor (LOF).
	Analisi delle anomalie identificate e loro confronto con eventi “normali”.
	
	- Obiettivo:
	Mostrare come tecniche avanzate di rilevamento delle anomalie possano essere applicate a eventi di fisica delle particelle per scoprire segnali rari o nuovi fenomeni.

5. **Confronto Statistico tra Eventi ad Alto e Basso MET**

	- Descrizione: 
	Esegui un confronto statistico tra eventi con alto MET e basso MET per vedere se ci sono differenze significative nelle altre variabili cinematiche (MR, Rsq, HT, nJets).
	
	- Approcci Statistici:
	Test statistici come il t-test per campioni indipendenti o il test di Mann-Whitney.
	Analisi delle differenze tra distribuzioni utilizzando test di Kolmogorov-Smirnov.
	Visualizzazione delle differenze tramite boxplot e violin plot.
	
	- Obiettivo:
	 Evidenziare differenze significative tra eventi ad alta e bassa energia trasversa mancante, suggerendo potenziali segnali di particelle invisibili o nuove particelle.

6. **Test d’Ipotesi per la Ricerca di Nuove Particelle Supersimmetriche**

	- Descrizione: 
	Testa l’ipotesi che alcuni eventi con valori estremi di MR e Rsq potrebbero essere dovuti a particelle supersimmetriche. Confronta la distribuzione osservata con quella attesa dal Modello Standard.
	
	- Approcci Statistici:
	Test di ipotesi, come il chi-quadrato o il likelihood ratio test.
	Simulazioni Monte Carlo per generare dati secondo le aspettative del Modello Standard e confrontarli con i dati osservati.
	
	- Obiettivo:
	Presentare un’analisi rigorosa delle variabili razor per verificare la possibilità di scoprire nuove particelle, confrontando dati reali con modelli teorici.


**Dataset Content**

1. **Run (Numero della corsa)**:
   - **Definizione**: Ogni evento di fisica delle particelle è registrato durante una specifica "run" o corsa, che rappresenta una serie di eventi registrati in un certo intervallo temporale. Questa variabile identifica quale corsa ha generato l'evento.
   - **Uso**: Ai fini analitici può essere utile per selezionare eventi da specifiche corse oppure per controllare la qualità dei dati in diverse corse.

2. **Lumi (Sezione di luminosità)**:
   - **Definizione**: Le corse sono suddivise in sezioni di luminosità, un'altra suddivisione temporale dei dati raccolti. La luminosità è direttamente collegata al numero di collisioni che si verificano in un dato periodo.
   - **Uso**: Può essere utile per filtrare o suddividere gli eventi a seconda di diverse sezioni di luminosità, ad esempio per verificare se ci sono fluttuazioni dovute a cambiamenti nelle condizioni sperimentali.

3. **Event (Numero dell'evento)**:
   - **Definizione**: Ogni evento è un’osservazione specifica, ovvero un singolo risultato da una collisione di particelle. Questo numero identifica univocamente ogni evento.
   - **Uso**: È un identificatore univoco e può essere utile per selezionare o escludere specifici eventi.

4. **MR (Prima variabile razor)**:
   - **Definizione**: MR è una variabile cinetica usata nelle ricerche di particelle supersimmetriche. Questa variabile stima una scala di massa complessiva e, nel limite di prodotti di decadimento senza massa, corrisponde alla massa della particella genitrice pesante.
   - **Interpretazione**: Valori elevati di MR potrebbero indicare la presenza di particelle pesanti nei decadimenti. In generale, MR è associata alla massa delle particelle prodotte nella collisione.

5. **Rsq (Seconda variabile razor)**:
   - **Definizione**: Rsq è la variabile quadrata di un rapporto (R) che misura il flusso di energia nel piano perpendicolare al fascio di particelle. Indica anche come il momento è distribuito tra le particelle visibili e quelle invisibili nell’evento.
   - **Interpretazione**: Valori più alti di Rsq suggeriscono un maggiore squilibrio nella distribuzione del momento trasversale, spesso associato alla presenza di particelle invisibili come neutrini o potenziali candidati di materia oscura.

6. (/9) **E1, Px1, Py1, Pz1 (Quattro-vettore del megajet principale)**:
   - **Definizione**: Rappresentano l'energia e le componenti di momento (in x, y e z) del megajet con il momento trasversale più alto. Un megajet è un gruppo di jets combinati in un singolo oggetto per semplificare l'analisi degli eventi complessi.
   - **Interpretazione**: Sono essenziali per ricostruire la dinamica dell'evento. Valori più alti di \(P_T\) indicano jets più energetici.

10. (/13) **E2, Px2, Py2, Pz2 (Quattro-vettore del megajet sub-leading)**:
   - **Definizione**: Questi quattro valori corrispondono all'energia e al momento del megajet con il secondo più alto momento trasversale nell'evento.
   - **Interpretazione**: Anche qui, valori elevati indicano jets particolarmente energetici e questi megajets possono fornire indicazioni sulla topologia dell'evento.

14. **HT (Somma scalare dei momenti trasversali)**:
   - **Definizione**: HT è la somma scalare del momento trasversale di tutti i jets nell’evento. È una variabile che riassume l'energia totale dei jets in un evento.
   - **Interpretazione**: Valori elevati di HT spesso indicano eventi altamente energetici, che possono essere associati a nuovi processi fisici come il decadimento di particelle pesanti.

15. **MET (Missing Transverse Energy, Energia trasversa mancante)**:
   - **Definizione**: MET rappresenta l'energia trasversale mancante in un evento, calcolata come la magnitudine della somma vettoriale delle energie trasversali di tutte le particelle visibili. Valori non nulli di MET indicano la presenza di particelle invisibili (come neutrini) o l'eventuale produzione di nuove particelle non rilevate.
   - **Interpretazione**: MET è una variabile cruciale nella ricerca di fisica oltre il Modello Standard, come la materia oscura. Alti valori di MET possono suggerire la presenza di particelle invisibili.

16. **nJets (Numero di jets)**:
   - **Definizione**: Conta il numero di jets con momento trasversale superiore a 40 GeV.
   - **Interpretazione**: Un alto numero di jets potrebbe indicare eventi complessi, come quelli associati a particelle pesanti che decadono in molti sottoprodotti.

17. **nBJets (Numero di jets b-tagged)**:
   - **Definizione**: Conta il numero di jets identificati come provenienti da quark b, con momento trasversale superiore a 40 GeV.
   - **Interpretazione**: I b-jets sono essenziali per identificare processi che coinvolgono quark pesanti, come la produzione di top quark o eventi associati alla fisica del bosone di Higgs.
