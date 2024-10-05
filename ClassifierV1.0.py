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
 
 
# Set style for better-looking plots
try:
    plt.style.use('ggplot')
except Exception as e:
    print(f"Warning: Could not set plot style. Error: {e}")
    print("Continuing with default style.")
 
# Use a color-blind friendly palette
sns.set_palette("colorblind")
 
# Load the data
df = pd.read_csv('/Users/aldojacopo/Library/CloudStorage/OneDrive-Uniparthenope/SIAFA 2 - Analisi Espolarativa/ProjectWork/MultiJetRun2010B.csv')
 
# Function for advanced descriptive statistics
def advanced_describe(data, var_name):
    stats = data[var_name].describe(percentiles=[.05, .25, .5, .75, .95])
    skewness = data[var_name].skew()
    kurtosis = data[var_name].kurtosis()
    
    print(f"\n{var_name}:")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['50%']:.4f}")
    print(f"Std Dev: {stats['std']:.4f}")
    print(f"Min: {stats['min']:.4f}")
    print(f"Max: {stats['max']:.4f}")
    print(f"5th Percentile: {stats['5%']:.4f}")
    print(f"95th Percentile: {stats['95%']:.4f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
 
# Calculate advanced descriptive statistics for all numerical variables
print("Advanced Descriptive Statistics:")
numerical_vars = df.select_dtypes(include=[np.number]).columns
for var in numerical_vars:
    advanced_describe(df, var)
 
# Correlation analysis
corr_matrix = df[numerical_vars].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)
 
# Enhanced visualizations
def plot_distribution(data, var_name):
    plt.figure(figsize=(12, 6))
    
    # Histogram with KDE
    sns.histplot(data[var_name], kde=True, stat="density", linewidth=0, alpha=0.7)
    
    # Add a more sophisticated density estimation
    sns.kdeplot(data[var_name], color="crimson", lw=2, label="KDE")
    
    # Add rug plot
    sns.rugplot(data[var_name], color="black", alpha=0.5)
    
    plt.title(f'Distribution of {var_name}', fontsize=16)
    plt.xlabel(var_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.savefig(f'distribution_{var_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
 
# Plot enhanced distributions for all numerical variables
for var in numerical_vars:
    plot_distribution(df, var)
 
# Enhanced correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Heatmap', fontsize=16)
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
 
# Pairplot for multivariate analysis (using a subset of variables to keep the plot readable)
subset_vars = ['MR', 'Rsq', 'HT', 'MET', 'nJets', 'nBJets']
sns.pairplot(df[subset_vars], diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pairwise Relationships', y=1.02, fontsize=16)
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
 
# Function to identify outliers using Isolation Forest
def identify_outliers_iforest(data, contamination=0.1):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(data)
    return outliers == -1
 
# Identify outliers
outliers = identify_outliers_iforest(df[numerical_vars])
print(f"\nNumber of potential outliers identified: {sum(outliers)}")
 
# Visualize outliers in 2D space using PCA
scaler = StandardScaler()
pca = PCA(n_components=2)
df_scaled = scaler.fit_transform(df[numerical_vars])
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
 
# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_scaled)
 
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('K-means Clustering Visualization (PCA)', fontsize=16)
plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
 
# Analyze relationship between categorical and numerical variables
categorical_vars = df.select_dtypes(include=['object']).columns
for cat_var in categorical_vars:
    plt.figure(figsize=(12, 10))
    for i, num_var in enumerate(subset_vars, 1):
        plt.subplot(3, 2, i)
        sns.boxplot(x=cat_var, y=num_var, data=df)
        sns.swarmplot(x=cat_var, y=num_var, data=df, color=".25", size=3, alpha=0.5)
        plt.title(f'{num_var} vs {cat_var}', fontsize=14)
        plt.xlabel(cat_var, fontsize=12)
        plt.ylabel(num_var, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'variables_vs_{cat_var}.png', dpi=300, bbox_inches='tight')
    plt.close()
 
print("\nAll visualizations have been saved as high-quality PNG files.")
 
# Additional statistical tests
for var in numerical_vars:
    _, p_value = stats.normaltest(df[var])
    print(f"\nNormality test for {var}:")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("The distribution is likely not normal.")
    else:
        print("The distribution might be normal.")
 
    # Q-Q plot
    plt.figure(figsize=(10, 6))
    qqplot(df[var], line='s')
    plt.title(f'Q-Q Plot for {var}', fontsize=16)
    plt.savefig(f'qq_plot_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()
 
# ANOVA and Tukey's HSD test for categorical variables
for cat_var in categorical_vars:
    for num_var in subset_vars:
        f_statistic, p_value = stats.f_oneway(*[group[num_var].values for name, group in df.groupby(cat_var)])
        print(f"\nANOVA test for {num_var} grouped by {cat_var}:")
        print(f"F-statistic: {f_statistic:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("There are significant differences between groups.")
            tukey_results = pairwise_tukeyhsd(df[num_var], df[cat_var])
            print("\nTukey's HSD Test Results:")
            print(tukey_results)
        else:
            print("No significant differences between groups.")
 
# Calculate and print explained variance ratio for PCA
explained_variance_ratio = pca.explained_variance_ratio_
print("\nPCA Explained Variance Ratio:")
print(f"First component: {explained_variance_ratio[0]:.4f}")
print(f"Second component: {explained_variance_ratio[1]:.4f}")
 
# Cumulative explained variance plot
plt.figure(figsize=(10, 6))
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.savefig('cumulative_variance_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
 
print("\nAdvanced Comprehensive EDA completed. Please review the generated visualizations and statistical outputs.")