import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class HeartDiseaseAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.output_dir = self.create_output_directory()

    def create_output_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"heart_disease_analysis_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def save_plot(self, plt, filename):
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def show_data_info(self):
        with open(os.path.join(self.output_dir, 'data_info.txt'), 'w') as f:
            f.write("Primi 5 righe del dataset:\n")
            f.write(self.df.head().to_string())
            f.write("\n\nInformazioni sul dataset:\n")
            self.df.info(buf=f)
            f.write("\n\nStatistiche descrittive:\n")
            f.write(self.df.describe().to_string())
            f.write("\n\nConteggio dei valori nulli:\n")
            f.write(self.df.isnull().sum().to_string())

    def encode_categorical_variables(self):
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_cols:
            self.df[col] = pd.Categorical(self.df[col]).codes

    def plot_target_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='num', data=self.df, palette='viridis')
        plt.title('Distribuzione della variabile target (num)', fontsize=16)
        plt.xlabel('Classe di malattia cardiaca', fontsize=12)
        plt.ylabel('Conteggio', fontsize=12)
        self.save_plot(plt, 'target_distribution.png')

    def plot_correlation_matrix(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(14, 12))
        sns.heatmap(self.df[numeric_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
        plt.title('Matrice di correlazione', fontsize=16)
        self.save_plot(plt, 'correlation_matrix.png')

    def plot_numeric_distributions(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_cols = 3
        n_rows = (len(numeric_cols) - 1) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(self.df[col], kde=True, ax=axes[i], color='skyblue', edgecolor='black')
            
            mean = self.df[col].mean()
            median = self.df[col].median()
            mode = self.df[col].mode().values[0]
            
            axes[i].axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Media: {mean:.2f}')
            axes[i].axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Mediana: {median:.2f}')
            axes[i].axvline(mode, color='purple', linestyle='dashed', linewidth=2, label=f'Moda: {mode:.2f}')
            
            skewness = self.df[col].skew()
            kurtosis = self.df[col].kurtosis()
            
            axes[i].set_title(f'Distribuzione di {col}\nSkewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}', fontsize=12)
            axes[i].legend(fontsize=10)

        plt.tight_layout()
        self.save_plot(plt, 'numeric_distributions.png')

    def plot_boxplot(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(16, 10))
        sns.boxplot(data=self.df[numeric_cols], palette='Set3')
        plt.title('Boxplot delle variabili numeriche', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        self.save_plot(plt, 'boxplot.png')

    def analyze_age_vs_heart_disease(self):
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='num', y='age', data=self.df, palette='Set2')
        plt.title('Relazione tra età e malattie cardiache', fontsize=16)
        plt.xlabel('Classe di malattia cardiaca', fontsize=12)
        plt.ylabel('Età', fontsize=12)
        self.save_plot(plt, 'age_vs_heart_disease.png')

    def perform_t_tests(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results = []
        for col in numeric_cols:
            if col != 'num':
                t_stat, p_value = stats.ttest_ind(self.df[self.df['num'] == 0][col], self.df[self.df['num'] == 1][col])
                results.append({
                    'Variable': col,
                    't-statistic': t_stat,
                    'p-value': p_value
                })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, 't_test_results.csv'), index=False)

    def analyze_categorical_vs_heart_disease(self):
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        chi_square_results = []
        
        for col in categorical_cols:
            plt.figure(figsize=(12, 7))
            sns.countplot(x=col, hue='num', data=self.df, palette='Set1')
            plt.title(f'Distribuzione di {col} per classe di malattia cardiaca', fontsize=16)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Conteggio', fontsize=12)
            plt.legend(title='Classe di malattia cardiaca', title_fontsize='12', fontsize='10')
            self.save_plot(plt, f'categorical_{col}_vs_heart_disease.png')

            contingency_table = pd.crosstab(self.df[col], self.df['num'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            chi_square_results.append({
                'Variable': col,
                'chi2': chi2,
                'p-value': p_value,
                'degrees_of_freedom': dof
            })
        
        chi_square_df = pd.DataFrame(chi_square_results)
        chi_square_df.to_csv(os.path.join(self.output_dir, 'chi_square_results.csv'), index=False)

    def handle_missing_values(self):
        # Impute numeric columns with median
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_imputer = SimpleImputer(strategy='median')
        self.df[numeric_columns] = numeric_imputer.fit_transform(self.df[numeric_columns])

        # Impute categorical columns with most frequent value
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.df[categorical_columns] = categorical_imputer.fit_transform(self.df[categorical_columns])

    def preprocess_data(self):
        self.handle_missing_values()
        
        numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ])
        
        X = self.df.drop('num', axis=1)
        y = self.df['num']
        
        X_preprocessed = preprocessor.fit_transform(X)
        
        # Create a new dataframe with preprocessed data
        feature_names = (numeric_features + 
                         [f"{feat}_{cat}" for feat, cats in zip(categorical_features, preprocessor.named_transformers_['cat'].categories_) 
                          for cat in cats[1:]])
        self.df_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
        self.df_preprocessed['num'] = y

    def run_analysis(self):
        print("Iniziando l'analisi...")
        self.show_data_info()
        print("Informazioni sui dati salvate in data_info.txt")
        
        self.preprocess_data()
        print("Dati preprocessati")

        # Use df_preprocessed for the following analyses
        self.plot_target_distribution()
        print("Grafico della distribuzione target salvato")
        
        self.plot_correlation_matrix()
        print("Matrice di correlazione salvata")
        
        self.plot_numeric_distributions()
        print("Grafici delle distribuzioni numeriche salvati")
        
        self.plot_boxplot()
        print("Boxplot salvato")
        
        self.analyze_age_vs_heart_disease()
        print("Analisi età vs malattie cardiache completata")
        
        self.perform_t_tests()
        print("T-test completati e risultati salvati in t_test_results.csv")
        
        self.analyze_categorical_vs_heart_disease()
        print("Analisi delle variabili categoriche completata e risultati salvati")
        
        print(f"Analisi completata. Tutti i risultati sono stati salvati nella cartella: {self.output_dir}")

# Esempio di utilizzo
if __name__ == "__main__":
    analysis = HeartDiseaseAnalysis('heart_disease_uci.csv')
    analysis.run_analysis()