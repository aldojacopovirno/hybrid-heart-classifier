**Obiettivi del Progetto**

Il progetto si propone di raggiungere due obiettivi principali, ciascuno legato a una diversa area della fisica delle particelle. Di seguito sono descritti in dettaglio:

1. Riconoscimento di Particelle
	Obiettivo Principale:
	- Sviluppare un sistema in grado di identificare e classificare diverse particelle (elettroni, protoni, pioni) a partire dalle tracce lasciate nei rivelatori. Questa classificazione si basa su variabili fisiche chiave come la traiettoria, l'impulso e il tempo di volo delle particelle.

	Obiettivi Specifici:
	- Identificazione delle tracce: Creare un algoritmo che esamini le tracce delle particelle nei rivelatori e determini il tipo di particella sulla base delle loro proprietà fisiche.
	- Precisione e Affidabilità: Migliorare l'accuratezza della classificazione utilizzando tecniche avanzate come il clustering e le reti neurali.
	- Visualizzazione dei Risultati: Fornire visualizzazioni chiare e intuitive per rappresentare le particelle identificate e le loro tracce nel rivelatore.

2. Rilevamento di Anomalie
	Obiettivo Principale: 
	- Utilizzare tecniche di anomaly detection per identificare eventi anomali nei dati delle collisioni, potenzialmente indicative di nuovi fenomeni fisici o di errori di misurazione.

	Obiettivi Specifici:
	- Rilevamento di Eventi Anomali: Implementare algoritmi che possono rilevare eventi non comuni o inattesi nei dati, che potrebbero indicare nuove interazioni o particelle.
	- Analisi di Eventi Rilevanti: Analizzare i risultati delle anomalie per determinare se rappresentano nuovi fenomeni fisici o semplici errori strumentali.
	- Applicazione di Metodologie Statistiche: Utilizzare metodi statistici avanzati per garantire che le anomalie siano correttamente identificate e valutate.


**DataFrame**

https://opendata.cern.ch/record/545 

Il DataFrame utilizzato nel progetto conterrà dati provenienti da esperimenti di fisica delle particelle, in particolare quelli di tracciamento delle particelle da esperimenti come ALICE o CMS. 

Struttura del DataFrame
Il DataFrame avrà la seguente struttura, comprendente diverse colonne che rappresentano variabili fisiche delle particelle:

| Colonna       | Descrizione                                                                 
|---------------|-----------------------------------------------------------------------------
| Run       | Numero del run dell'evento.                                              
| Event     | Numero dell'evento.                                                      
| E         | Energia totale della particella (in GeV).                                 
| px, py, pz| Componenti del momento della particella (in GeV).                        
| pt        | Momento trasversale della particella (in GeV).                          
| eta       | Pseudorapidity della particella.                                          
| phi       | Angolo phi (in radianti) della particella.                                
| Q         | Carica della particella.                                                  
| M         | Massa invariata (in GeV) di coppie di particelle.                        
| chiSq     | Chi-quadro per grado di libertà della particella.                        
| dxy       | Parametro di impatto nel piano trasversale rispetto al vertice.           
| iso       | Isolation combinata della particella.                                     
| MET       | Momento trasversale mancante dell'evento (in GeV).                       
| phiMET    | Angolo phi del momento trasversale mancante (in radianti).                
| type      | Tipo di particella (ad esempio, EB, EE per elettroni; T, G per muoni).  
| delEta    | Differenza in eta tra il tracciato della particella e il cluster ECAL. 
| delPhi    | Differenza in phi tra il tracciato della particella e il cluster ECAL. 
| sigmaEtaEta| RMS del cluster lungo eta per un elettrone.                            
| HoverE    | Rapporto tra l'energia nel HCAL e quella nel ECAL per un elettrone.    
| isoTrack  | Variabile di isolamento per l'elettrone nel tracciato.             
| isoEcal   | Variabile di isolamento per l'elettrone nell'ECAL.                    
| isoHcal   | Variabile di isolamento per l'elettrone nell'HCAL.                       

I dati saranno estratti dai seguenti file CSV, ognuno rappresentante un diverso tipo di evento:
- Wmunu.csv: Contiene dati per eventi W+μ (muonico).
- Wenu.csv: Contiene dati per eventi W+e (elettronico).
- Zmumu.csv: Contiene dati per eventi Z+μ (muonico).



**Assunzioni Fisiche di Base**

1.	Teoria dei Campi: I dati analizzati provengono da esperimenti in fisica delle particelle, dove si assume che le interazioni tra particelle subatomiche possano essere descritte tramite la teoria quantistica dei campi. Gli eventi di collisione generano particelle secondo le previsioni del Modello Standard.
2.	Conservazione della Momento: Si assume che in ogni evento di collisione la quantità di moto totale sia conservata. Questo principio è cruciale per calcolare le proprietà delle particelle generate, inclusi i leptoni.
3.	Proprietà delle Particelle: Le particelle selezionate (elettroni e muoni) seguono traiettorie prevedibili in un campo magnetico, il che consente di determinare il loro momento e le altre proprietà attraverso l’analisi delle loro tracce nel rivelatore.
4.	Selezione di Eventi: Gli eventi sono stati selezionati secondo criteri specifici (ad esempio, soglie di momento trasversale), riflettendo l’intento di catturare eventi rilevanti per le ricerche su W e Z bosoni.
5.	Isolamento delle Particelle: Le variabili di isolamento calcolate (es. iso, isoTrack, isoEcal, isoHcal) sono utilizzate per differenziare tra eventi significativi e sfondi di rumore.

**Obiettivi Finali**
1.	Identificazione e Classificazione delle Particelle:
	Sviluppare un sistema per identificare e classificare eventi di collisione in base alle proprietà fisiche delle particelle coinvolte (elettroni, muoni, etc.). Utilizzare tecniche di clustering e classificazione per ottenere una categorizzazione accurata degli eventi.
2.	Rilevamento di Anomalie:
	Implementare tecniche di anomaly detection per identificare eventi insoliti che potrebbero indicare nuovi fenomeni fisici o errori di misurazione. Ciò potrebbe aiutare a rivelare potenziali scoperte nel campo della fisica delle particelle.
3.	Visualizzazione e Interpretazione dei Dati:
	Creare visualizzazioni informative per rappresentare i dati analizzati, facilitando l’interpretazione dei risultati e la comunicazione con la comunità scientifica e il pubblico.

**Pipeline**
 
1. Acquisizione e pre-elaborazione dei dati
	-Pulizia dei dati:
		Rimozione di dati mancanti: Identificare e rimuovere osservazioni incomplete o corrotte che potrebbero falsare il risultato.
	Gestione del rumore: 
		Implementare tecniche di smoothing (come un filtro gaussiano o mediano) per ridurre il rumore nei dati di tracciamento.
	- Feature engineering:
		Calcolare feature rilevanti come velocità, angoli di deviazione, distanze percorse, massa stimata, etc. Si possono anche costruire feature derivate dalla combinazione di altre proprietà, come l’energia e il momento angolare.
		Utilizzare PCA (Principal Component Analysis) per ridurre la dimensionalità del dataset e conservare solo le componenti principali, semplificando l’analisi.

2. Clustering e segmentazione delle tracce
	- K-means:
		Applicare il clustering K-means per segmentare le tracce in gruppi distinti basati sulle loro caratteristiche.
		Si consiglia di testare più valori di k utilizzando l’algoritmo “Elbow method” per scegliere il numero ottimale di cluster.
	- DBSCAN:
		Utilizzare DBSCAN (Density-Based Spatial Clustering of Applications with Noise) per identificare gruppi di particelle simili e separare gli outliers, che potrebbero essere eventi anomali.

3. Classificazione delle particelle
	- Algoritmi supervisati: 
		Dopo il clustering preliminare, si esegue la classificazione supervisionata per identificare il tipo specifico di particella (elettrone, protone, pion) sulla base delle loro caratteristiche fisiche.
	- Support Vector Machines (SVM):
	Si può utilizzare SVM per separare le particelle in classi in base alle feature precedentemente ingegnerizzate.
	- Random Forest:
	Implementare un Random Forest per una classificazione più robusta, in particolare se i dati sono sbilanciati. Questo metodo può gestire bene le variazioni e le complessità del dataset.

4. Rilevamento delle anomalie
	- Isolation Forest: 
		Utilizzare questo algoritmo per isolare eventi rari che non rientrano nella distribuzione statistica dei dati normali.
	- Autoencoders: 
		Implementare una rete neurale autoencoder non supervisionata che cerca di ricostruire i dati delle particelle. Gli errori di ricostruzione elevati indicano anomalie.
	- DBSCAN: 
		Riutilizzare DBSCAN come metodo di rilevamento di anomalie, grazie alla sua capacità di identificare outliers non appartenenti ad alcun cluster.

5. Validazione e interpretazione dei risultati
	- Cross-validation: 
		Implementare tecniche di cross-validation per valutare l’accuratezza e la robustezza dei modelli. Misurare metriche come precision, recall e F1-score.
	- Analisi delle anomalie: 
		Gli eventi segnalati come anomali dal sistema dovrebbero essere sottoposti a un’analisi fisica dettagliata per determinare se sono il risultato di fenomeni fisici nuovi o errori.
