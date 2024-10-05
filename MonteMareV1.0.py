import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Caricamento dei dati
def load_data(file_paths):
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

file_paths = ['Wmunu.csv', 'Wenu.csv', 'Zmumu.csv']
data = load_data(file_paths)

# 2. Pulizia dei dati
def clean_data(df):
    # Rimozione di righe con valori mancanti
    df_cleaned = df.dropna()
    
    # Rimozione di duplicati
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned

data_cleaned = clean_data(data)

# 3. Gestione del rumore (esempio con un filtro mediano)
def apply_median_filter(df, columns, window_size=3):
    for col in columns:
        df[col] = df[col].rolling(window=window_size, center=True).median()
    return df

columns_to_filter = ['E', 'px', 'py', 'pz', 'pt']
data_filtered = apply_median_filter(data_cleaned, columns_to_filter)

# 4. Feature engineering
def engineer_features(df):
    # Calcolo della velocità (assumendo c=1)
    df['velocity'] = df['E'] / np.sqrt(df['E']**2 - (df['px']**2 + df['py']**2 + df['pz']**2))
    
    # Calcolo dell'angolo di deviazione rispetto all'asse z
    df['deviation_angle'] = np.arccos(df['pz'] / np.sqrt(df['px']**2 + df['py']**2 + df['pz']**2))
    
    # Calcolo della distanza percorsa nel piano trasversale
    df['transverse_distance'] = np.sqrt(df['px']**2 + df['py']**2)
    
    # Stima della massa (usando E^2 = p^2 + m^2)
    df['estimated_mass'] = np.sqrt(df['E']**2 - (df['px']**2 + df['py']**2 + df['pz']**2))
    
    return df

data_engineered = engineer_features(data_filtered)

# 5. Normalizzazione dei dati
def normalize_data(df):
    scaler = StandardScaler()
    columns_to_normalize = ['E', 'px', 'py', 'pz', 'pt', 'eta', 'phi', 'M', 'chiSq', 'dxy', 'iso', 'MET',
                            'velocity', 'deviation_angle', 'transverse_distance', 'estimated_mass']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

data_normalized = normalize_data(data_engineered)

# 6. Applicazione di PCA
def apply_pca(df, n_components=0.95):
    pca = PCA(n_components=n_components)
    columns_for_pca = [col for col in df.columns if col not in ['Run', 'Event', 'Q', 'type']]
    pca_result = pca.fit_transform(df[columns_for_pca])
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    
    # Aggiungiamo le colonne non utilizzate per PCA al dataframe risultante
    for col in ['Run', 'Event', 'Q', 'type']:
        pca_df[col] = df[col]
    
    return pca_df

data_pca = apply_pca(data_normalized)

# Salvataggio dei dati preprocessati
data_pca.to_csv('preprocessed_particle_data.csv', index=False)

print("Preprocessing completato. I dati sono stati salvati in 'preprocessed_particle_data.csv'.")
print(f"Dimensioni del dataset originale: {data.shape}")
print(f"Dimensioni del dataset preprocessato: {data_pca.shape}")