import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Loading
def load_data(file_paths):
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

file_paths = ['Wmunu.csv', 'Wenu.csv', 'Zmumu.csv']
data = load_data(file_paths)

# 2. Exploratory Data Analysis (EDA)
def perform_eda(df):
    print(df.describe())
    
    # Correlation matrix
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Distribution of main variables
    num_cols = ['E', 'pt', 'eta', 'phi', 'M', 'MET']
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i//3, i%3])
    plt.tight_layout()
    plt.savefig('distributions.png')
    plt.close()

perform_eda(data)

# 3. Outlier Handling
def handle_outliers(df):
    # Method 1: Removal based on Z-score
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df_no_outliers = df[(z_scores < 3).all(axis=1)]
    
    # Method 2: Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = iso_forest.fit_predict(df.select_dtypes(include=[np.number]))
    df_no_outliers = df[outlier_labels != -1]
    
    return df_no_outliers

data_cleaned = handle_outliers(data)

# 4. Advanced Missing Value Imputation
def impute_missing_values(df):
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

data_imputed = impute_missing_values(data_cleaned)

# 5. Advanced Feature Engineering
def advanced_feature_engineering(df):
    # Calculate missing transverse energy
    df['MET_T'] = df['MET'] * np.cos(df['phiMET'])
    
    # Energy to momentum ratio
    df['E_p_ratio'] = df['E'] / np.sqrt(df['px']**2 + df['py']**2 + df['pz']**2)
    
    # Energy distribution asymmetry
    df['energy_asymmetry'] = (df['E'] - df['pt']) / (df['E'] + df['pt'])
    
    # Angular momentum
    df['angular_momentum'] = df['pt'] * df['eta']
    
    return df

data_engineered = advanced_feature_engineering(data_imputed)

# 6. Robust Normalization
def robust_normalize(df):
    scaler = RobustScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

data_normalized = robust_normalize(data_engineered)

# 7. Principal Component Analysis (PCA) with Automatic Selection
def apply_pca_auto(df, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    pca_result = pca.fit_transform(df[numeric_columns])
    
    # Calculate the number of components that explain the desired variance
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_threshold) + 1
    
    pca_df = pd.DataFrame(data=pca_result[:, :n_components], 
                          columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Add categorical columns to the resulting dataframe
    for col in df.select_dtypes(include=['object']).columns:
        pca_df[col] = df[col]
    
    print(f"Number of principal components selected: {n_components}")
    print("Explained variance ratio per component:")
    print(pca.explained_variance_ratio_[:n_components])
    
    return pca_df

data_pca = apply_pca_auto(data_normalized)

# 8. Basic Statistical Analysis by Particle Class
def basic_statistical_analysis(df):
    if 'type' not in df.columns:
        print("The 'type' column is not present in the dataset. Make sure you have a column that identifies the particle type.")
        return

    numeric_features = ['E', 'pt', 'eta', 'phi', 'M', 'MET', 'chiSq', 'dxy', 'iso']

    stats_by_class = {}

    for particle_type in df['type'].unique():
        particle_data = df[df['type'] == particle_type]
        
        class_stats = {}
        for feature in numeric_features:
            feature_data = particle_data[feature].dropna()
            
            class_stats[feature] = {
                'count': len(feature_data),
                'mean': np.mean(feature_data),
                'median': np.median(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'skewness': stats.skew(feature_data),
                'kurtosis': stats.kurtosis(feature_data)
            }
        
        stats_by_class[particle_type] = class_stats

    stats_df = pd.DataFrame({(outerKey, innerKey): values 
                             for outerKey, innerDict in stats_by_class.items() 
                             for innerKey, values in innerDict.items()})

    stats_df.to_csv('particle_class_statistics.csv')

    print("Basic statistical analysis completed. Results have been saved in 'particle_class_statistics.csv'")

    for feature in numeric_features:
        plt.figure(figsize=(12, 6))
        for particle_type in df['type'].unique():
            sns.kdeplot(df[df['type'] == particle_type][feature], label=particle_type)
        plt.title(f'Distribution of {feature} by Particle Type')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'distribution_{feature}.png')
        plt.close()

    print("Distribution plots have been saved as PNG files.")

    return stats_by_class

# Execute basic statistical analysis
basic_stats = basic_statistical_analysis(data)

# 9. Save preprocessed data
data_pca.to_csv('advanced_preprocessed_particle_data.csv', index=False)

print("Advanced preprocessing and basic statistical analysis completed.")
print("Basic statistics have been saved in 'particle_class_statistics.csv'")
print("Preprocessed data has been saved in 'advanced_preprocessed_particle_data.csv'")
print(f"Original dataset dimensions: {data.shape}")
print(f"Preprocessed dataset dimensions: {data_pca.shape}")