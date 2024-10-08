import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class HeartDiseaseAnalysis:
    def __init__(self, file_path):
        # Caricamento dei dati
        self.df = pd.read_csv(file_path)

    def show_data_info(self):
        # Visualizzazione delle prime righe e informazioni sul dataset
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())
        print(self.df.isnull().sum())

    def encode_categorical_variables(self):
        # Conversione delle variabili categoriche
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_cols:
            self.df[col] = pd.Categorical(self.df[col]).codes

    def plot_target_distribution(self):
        # Distribuzione della variabile target
        plt.figure(figsize=(10, 6))
        sns.countplot(x='num', data=self.df)
        plt.title('Distribuzione della variabile target (num)')
        plt.show()

    def plot_correlation_matrix(self):
        # Correlazione tra le variabili numeriche
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df[numeric_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Matrice di correlazione')
        plt.show()

    def plot_numeric_distributions(self):
        # Distribuzione delle variabili numeriche
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_cols = 3
        n_rows = (len(numeric_cols) - 1) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(self.df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribuzione di {col}')

        plt.tight_layout()
        plt.show()

    def plot_boxplot(self):
        # Boxplot per identificare outliers
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(15, 10))
        self.df[numeric_cols].boxplot()
        plt.title('Boxplot delle variabili numeriche')
        plt.xticks(rotation=45)
        plt.show()

    def analyze_age_vs_heart_disease(self):
        # Analisi bivariata: relazione tra età e malattie cardiache
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='num', y='age', data=self.df)
        plt.title('Relazione tra età e malattie cardiache')
        plt.show()

    def perform_t_tests(self):
        # Test statistici tra le variabili numeriche e la variabile target
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'num':
                t_stat, p_value = stats.ttest_ind(self.df[self.df['num'] == 0][col], self.df[self.df['num'] == 1][col])
                print(f"T-test per {col}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    def analyze_categorical_vs_heart_disease(self):
        # Analisi delle variabili categoriche e la variabile target
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, hue='num', data=self.df)
            plt.title(f'Distribuzione di {col} per classe di malattia cardiaca')
            plt.show()

            # Chi-square test
            contingency_table = pd.crosstab(self.df[col], self.df['num'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            print(f"Chi-square test per {col}: chi2 = {chi2:.4f}, p-value = {p_value:.4f}")

# Esempio di utilizzo
if __name__ == "__main__":
    analysis = HeartDiseaseAnalysis('heart_disease_uci.csv')
    analysis.show_data_info()
    analysis.encode_categorical_variables()
    analysis.plot_target_distribution()
    analysis.plot_correlation_matrix()
    analysis.plot_numeric_distributions()
    analysis.plot_boxplot()
    analysis.analyze_age_vs_heart_disease()
    analysis.perform_t_tests()
    analysis.analyze_categorical_vs_heart_disease()