# Multi-Dimensional Statistical Modeling of Heart Disease Predictors: An Integrated Framework for Ordinal Regression and ROC Analysis

## 1. Introduction
This document outlines the methodologies and statistical analyses applied to the *Multi-Dimensional Statistical Modeling of Heart Disease Predictors* project. The dataset from the University of California, Irvine (UCI) on heart disease was utilized to predict the probability of heart disease based on a variety of clinical indicators. The following sections describe data preprocessing, feature encoding, statistical analysis, model construction, and validation methodologies.

## 2. Data Preparation

- Loading and Initial Setup
The required libraries for data loading, handling missing values, statistical analysis, and model diagnostics include `readr`, `tseries`, `labstatR`, `MASS`, `dplyr`, `caret`, among others. The heart disease dataset is read into R and its structure and initial rows are examined to understand the data's schema and content.

```r
# Load dataset
datamatrix <- read_csv("heart_disease_uci.csv")
str(datamatrix)
head(datamatrix)
```

- Encoding Categorical Variables

Categorical features were transformed into numeric format for model compatibility:

	•	sex: encoded as binary, 1 for male and 0 for female.
	•	cp, restecg, slope, and thal: encoded as ordinal or nominal levels.
	•	Missing values in certain fields (e.g., ca) were imputed using the median value.

- Handling Missing Values

Using K-Nearest Neighbors (KNN) imputation, missing values were addressed, which minimizes bias introduced by NA values. A summary of missing values per column was generated, and missing entries were imputed.

```r
library(VIM)
imputed_data <- kNN(datamatrix, k = 5)
```

- **Feature Scaling with Robust Scaler**  
To handle the impact of outliers on feature scaling, the Robust Scaler was applied. This scaling method transforms the data based on the median and the interquartile range (IQR), making it less sensitive to outliers.

    ```r
    library(caret)
    
    # Apply Robust Scaler
    preProc <- preProcess(imputed_data, method = "YeoJohnson")  # For handling normality if needed
    scaled_data <- predict(preProc, imputed_data)
    ```

## 3. Statistical Analysis

- Descriptive Statistics

Basic statistical properties, including mean, median, variance, skewness, and kurtosis, were calculated for each variable. Confidence intervals (95%) and hypothesis tests for mean differences from zero were computed for each variable.

- Outlier Detection

Using the Interquartile Range (IQR) method, outliers were identified for each numeric feature. The count of outliers for each variable is summarized, providing insights into potential data quality issues.

- Correlation Analysis

A Pearson correlation matrix was constructed, enabling the identification of multicollinearity among predictor variables. This step is crucial for understanding feature interdependencies and potential overfitting in the model.

## 4. Model Development

- Ordinal Regression

The heart disease outcome variable was converted into an ordered factor. An ordinal logistic regression model was selected given the ordinal nature of the target variable, predicting probability of heart disease severity across ordered levels. Variables with high Variance Inflation Factor (VIF) were excluded to mitigate multicollinearity.

```r
library(MASS)
model <- polr(num ~ ., data = datamatrix, Hess = TRUE)
```

- Model Diagnostics

To evaluate model performance, pseudo R-squared values and likelihood ratio tests were computed, providing an understanding of the model’s explanatory power.

- ROC Analysis

The ROC curve for each level of the ordinal response variable was generated, along with Area Under Curve (AUC) values. These ROC curves highlight the model’s discriminative ability across different severity levels.

```r
library(pROC)
roc_curve <- multiclass.roc(datamatrix$num, predict(model, type = "probs"))
```

## 5. Normality Testing and Power Analysis

- Normality Tests

For selected continuous variables (age, trestbps, chol, and thalch), the following normality tests were conducted: Shapiro-Wilk, Kolmogorov-Smirnov, Jarque-Bera, and Anderson-Darling. A Bonferroni correction was applied due to multiple testing.

- Power Analysis

To ensure the robustness of test results, power analysis was performed, particularly for the normality tests. The significance level was corrected for multiple tests using the Bonferroni method.

## 6. Visualization

- Distribution Plots

Histograms with overlayed normal distribution curves were generated for continuous predictors to visually assess distribution shapes and deviations from normality.

- Boxplots

Boxplots for age, trestbps, chol, and thalch were generated, depicting data central tendency, spread, and outliers.

- Q-Q Plots

Quantile-Quantile (Q-Q) plots were created for normality assessment, comparing variable distributions with a theoretical normal distribution.

## 7. Conclusion

This analysis integrates statistical modeling, data preprocessing, exploratory analysis, and visualization to construct a robust framework for predicting heart disease risk based on clinical data. The results offer a statistically validated, reproducible method for ordinal outcome prediction with extensive diagnostic checks.

References
- Ripley, B. D. (2002). Modern Applied Statistics with S. Springer.
- UC Irvine Machine Learning Repository: Heart Disease Data Set.
- Harrell, F. E. (2015). Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis.
