import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot
 
class DataAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.numerical_vars = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_vars = self.df.select_dtypes(include=['object']).columns
        self.subset_vars = ['MR', 'Rsq', 'HT', 'MET', 'nJets', 'nBJets']
        
        plt.style.use('ggplot')
        sns.set_palette("colorblind")
 
    def advanced_describe(self, var_name):
        stats = self.df[var_name].describe(percentiles=[.05, .25, .5, .75, .95])
        skewness = self.df[var_name].skew()
        kurtosis = self.df[var_name].kurtosis()
        
        return f"{var_name}:\n" \
               f"Mean: {stats['mean']:.4f}\n" \
               f"Median: {stats['50%']:.4f}\n" \
               f"Std Dev: {stats['std']:.4f}\n" \
               f"Min: {stats['min']:.4f}\n" \
               f"Max: {stats['max']:.4f}\n" \
               f"5th Percentile: {stats['5%']:.4f}\n" \
               f"95th Percentile: {stats['95%']:.4f}\n" \
               f"Skewness: {skewness:.4f}\n" \
               f"Kurtosis: {kurtosis:.4f}\n"
 
    def plot_distributions(self):
        fig, axes = plt.subplots(nrows=(len(self.numerical_vars) + 1) // 2, ncols=2, figsize=(20, 5 * len(self.numerical_vars)))
        axes = axes.flatten()
 
        for i, var in enumerate(self.numerical_vars):
            sns.histplot(self.df[var], kde=True, stat="density", linewidth=0, alpha=0.7, ax=axes[i])
            sns.kdeplot(self.df[var], color="crimson", lw=2, label="KDE", ax=axes[i])
            sns.rugplot(self.df[var], color="black", alpha=0.5, ax=axes[i])
            axes[i].set_title(f'Distribution of {var}')
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('Density')
            axes[i].legend()
 
        plt.tight_layout()
        plt.savefig('all_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
 
    def plot_correlation_heatmap(self):
        corr_matrix = self.df[self.numerical_vars].corr()
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Correlation Heatmap', fontsize=16)
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
 
    def plot_pairplot(self):
        sns.pairplot(self.df[self.subset_vars], diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle('Pairwise Relationships', y=1.02, fontsize=16)
        plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
 
    def identify_outliers(self, contamination=0.1):
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(self.df[self.numerical_vars])
        return outliers == -1
 
    def plot_outliers(self, outliers):
        scaler = StandardScaler()
        pca = PCA(n_components=2)
        df_scaled = scaler.fit_transform(self.df[self.numerical_vars])
        df_pca = pca.fit_transform(df_scaled)
 
        plt.figure(figsize=(10, 8))
        plt.scatter(df_pca[~outliers, 0], df_pca[~outliers, 1], c='blue', label='Normal', alpha=0.5)
        plt.scatter(df_pca[outliers, 0], df_pca[outliers, 1], c='red', label='Outlier', alpha=0.5)
        plt.title('PCA Visualization of Outliers', fontsize=16)
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.legend()
        plt.savefig('outliers_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
 
        return df_scaled, pca
 
    def plot_kmeans_clusters(self, df_scaled, pca):
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(df_scaled)
        df_pca = pca.transform(df_scaled)
 
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('K-means Clustering Visualization (PCA)', fontsize=16)
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
 
    def plot_categorical_vs_numerical(self):
        for cat_var in self.categorical_vars:
            plt.figure(figsize=(12, 10))
            for i, num_var in enumerate(self.subset_vars, 1):
                plt.subplot(3, 2, i)
                sns.boxplot(x=cat_var, y=num_var, data=self.df)
                sns.swarmplot(x=cat_var, y=num_var, data=self.df, color=".25", size=3, alpha=0.5)
                plt.title(f'{num_var} vs {cat_var}', fontsize=14)
                plt.xlabel(cat_var, fontsize=12)
                plt.ylabel(num_var, fontsize=12)
            plt.tight_layout()
            plt.savefig(f'variables_vs_{cat_var}.png', dpi=300, bbox_inches='tight')
            plt.close()
 
    def plot_qqplots(self):
        fig, axes = plt.subplots(nrows=(len(self.numerical_vars) + 1) // 2, ncols=2, figsize=(20, 5 * len(self.numerical_vars)))
        axes = axes.flatten()
 
        for i, var in enumerate(self.numerical_vars):
            qqplot(self.df[var], line='s', ax=axes[i])
            axes[i].set_title(f'Q-Q Plot for {var}')
 
        plt.tight_layout()
        plt.savefig('all_qqplots.png', dpi=300, bbox_inches='tight')
        plt.close()
 
    def perform_statistical_tests(self):
        results = []
        for var in self.numerical_vars:
            _, p_value = stats.normaltest(self.df[var])
            results.append(f"Normality test for {var}:\n"
                           f"p-value: {p_value:.4f}\n"
                           f"{'The distribution is likely not normal.' if p_value < 0.05 else 'The distribution might be normal.'}\n")
 
        for cat_var in self.categorical_vars:
            for num_var in self.subset_vars:
                f_statistic, p_value = stats.f_oneway(*[group[num_var].values for name, group in self.df.groupby(cat_var)])
                results.append(f"\nANOVA test for {num_var} grouped by {cat_var}:\n"
                               f"F-statistic: {f_statistic:.4f}\n"
                               f"p-value: {p_value:.4f}\n")
                
                if p_value < 0.05:
                    results.append("There are significant differences between groups.\n")
                    tukey_results = pairwise_tukeyhsd(self.df[num_var], self.df[cat_var])
                    results.append(f"Tukey's HSD Test Results:\n{tukey_results}\n")
                else:
                    results.append("No significant differences between groups.\n")
 
        return "\n".join(results)
 
    def plot_cumulative_variance_ratio(self, pca):
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
 
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
        plt.savefig('cumulative_variance_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()
 
        return f"PCA Explained Variance Ratio:\n" \
               f"First component: {explained_variance_ratio[0]:.4f}\n" \
               f"Second component: {explained_variance_ratio[1]:.4f}\n"
 
    def run_analysis(self):
        print("Advanced Descriptive Statistics:")
        for var in self.numerical_vars:
            print(self.advanced_describe(var))
 
        print("\nGenerating visualizations...")
        self.plot_distributions()
        self.plot_correlation_heatmap()
        self.plot_pairplot()
 
        outliers = self.identify_outliers()
        print(f"\nNumber of potential outliers identified: {sum(outliers)}")
        df_scaled, pca = self.plot_outliers(outliers)
 
        self.plot_kmeans_clusters(df_scaled, pca)
        self.plot_categorical_vs_numerical()
        self.plot_qqplots()
 
        print("\nPerforming statistical tests...")
        print(self.perform_statistical_tests())
 
        print("\nCalculating PCA explained variance ratio...")
        print(self.plot_cumulative_variance_ratio(pca))
 
        print("\nAdvanced Comprehensive EDA completed. Please review the generated visualizations and statistical outputs.")
 
if __name__ == "__main__":
    analyzer = DataAnalyzer('MultiJetRun2010B.csv')
    analyzer.run_analysis()