## Optimizing Heart Disease Classification Through Statistical-Mathematical Hybridization: An Integrated Ordinal-XGBoost Framework Analysis

This repository provides a comprehensive framework for heart disease classification and severity prediction using an hybrid model integrating ordinal logistic regression with XGBoost, alongside robust statistical analysis and ROC evaluations. This hybrid approach is designed to leverage both statistical rigor and machine learning capabilities, aiming for precise prediction and interpretability.


## Repository Structure

```R
CardioSTAT/
├── R/
│   ├── preprocessing.R        # Data preprocessing functions
│   ├── statistical_tests.R    # Implementation of statistical tests
│   ├── visualization.R        # Visualization functions
│   └── modeling.R             # Predictive models and regression
├── data/
│   └── heart_disease_uci.csv  # Raw data
├── scripts/
│   └── main_analysis.R        # Main analysis script
├── output/
│   ├── figures/               # Graphical outputs
│   └── results/               # Analysis results
├── docs/
│   └── methodology.md         # Methodology documentation
├── LICENSE
└── README.md
```

## Key Features

- **Robust Data Preprocessing** with KNN imputation and robust scaling
- **Hybrid Modeling Framework** combining Ordinal Logistic Regression and XGBoost
- **Advanced Descriptive Statistics**
- **Normality and Multicollinearity Testing** with Bonferroni Correction
- **Comprehensive Visualizations** (e.g., histograms, boxplots, Q-Q plots)
- **Class-Specific ROC Analysis** with multi-class AUC calculations
- **Automated Feature Selection** using XGBoost’s feature importance
- **Post-Model Diagnostics** including VIF and power analysis

## Prerequisites

The following R libraries are required:

```R
- readr        # CSV file reading
- labstatR     # Basic statistical functions
- tseries      # Time series analysis
- moments      # Skewness and kurtosis calculation
- VIM          # Missing data handling
- gridExtra    # Layout for graphical outputs
- lmtest       # Jarque-Bera test
- nortest      # Normality tests
- MASS         # Ordinal logistic regression
- car          # Model diagnostics
- olsrr        # Model diagnostics
- pscl         # Pseudo R-squared
- pwr          # Statistical power analysis
- dplyr        # Data manipulation
- caret        # Machine Learning and data preprocessing
- pROC         # ROC curves
- brant        # Proportional odds test for ordinal models
- xgboost      # Advanced machine learning models
```

## Getting Started

1. Clone the repository:

```R
git clone https://github.com/cardio-stats/CardioSTAT.git
```

2.	Run the setup script:

```R
source("scripts/setup.R")
```

3.	Run the complete analysis:

```R
source("scripts/main_analysis.R")
results <- run_complete_analysis("data/raw/heart_disease_uci.csv")
```

## Analysis Pipeline

1. Data Preprocessing (R/processing.R)
  - Encoding categorical variables
  - Handling missing values with KNN imputation
  - Applying Robust Scaler for feature scaling
  - Preparing data for ordinal regression

2. Basic Statistical Analysis (R/statistical_tests.R)
  - Comprehensive descriptive statistics
  - Confidence intervals
  - Hypothesis testing
  - Outlier analysis
  - Correlation matrices

3. Normality Tests (R/processing.R)
  - Shapiro-Wilk test
  - Kolmogorov-Smirnov test
  - Jarque-Bera test
  - Anderson-Darling test
  - Bonferroni correction for multiple tests
  - Statistical power analysis

4. Visualizations (R/visualization.R)
  - Histograms with normal density curves
  - Boxplots for outlier identification
  - Q-Q plots for normality assessment

5. Predictive Modeling (R/modeling.R)
   - Ordinal Logistic Regression for disease severity prediction
   - XGBoost for enhanced classification accuracy and feature
   - VIF analysis for multicollinearity
   - Model evaluation metrics including pseudo R-squared and AUC

6. Post-Model Diagnostics and Validation (R/modeling.R)
   - Diagnostic plots for model validation
   - Assessment of multicollinearity and feature importance with XGBoost
   - Statistical power analysis for model stability

7. ROC Analysis (R/modeling.R)
  - ROC curves for each class
  - AUC calculation per class
  - Comparative performance visualization

## Dataset Structure

The dataset should be saved in data/raw/ and include the following variables:

	•	age: Age of the patient
	•	sex: Gender of the patient
	•	cp: Type of chest pain
	•	trestbps: Resting blood pressure
	•	chol: Cholesterol
	•	fbs: Fasting blood sugar
	•	restecg: Resting electrocardiographic results
	•	thalch: Maximum heart rate
	•	exang: Exercise-induced angina
	•	oldpeak: ST depression
	•	slope: Slope of the ST segment
	•	ca: Number of major vessels
	•	thal: Thalassemia
	•	num: Heart disease diagnosis (target variable)

## Citation

If you use this software in your research, please cite:
```R
@software{CardioSTAT2024,
  author = {Virno, Aldo Jacopo and Bucchignani, Andrea},
  title =  {CardioSTAT: Optimizing Heart Disease Classification Through Statistical-Mathematical Hybridization},
  year = {2024},
  url = {https://github.com/cardio-stats/CardioSTAT}
}
```

## License

Distributed under the MIT License. See LICENSE for more information.

## Contact

- Aldo Jacopo Virno - aldojacopo@gmail.com
- Andrea Bucchignani - andreabucchignani@gmail.com
- Project Link: https://github.com/cardio-stats/CardioSTAT
