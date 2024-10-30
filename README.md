## Multi-Dimensional Statistical Modeling of Heart Disease Predictors: An Integrated Framework for Ordinal Regression and ROC Analysis

An integrated framework for statistical analysis and prediction of heart diseases using R, with a focus on ordinal regression and ROC analysis.

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

- **Comprehensive Data Preprocessing**
- **Descriptive Statistical Analysis**
- **Multiple Normality Tests with Bonferroni Correction**
- **Statistical Visualizations** (histograms, boxplots, Q-Q plots)
- **Ordinal Regression** for disease severity prediction
- **ROC Analysis** with curves for each class
- **Automated Handling of Missing Values** using KNN imputation
- **Feature Scaling with Robust Scaler** to mitigate the impact of outliers

## Prerequisites

The following R libraries are required:

```R
- readr      # CSV file reading
- labstatR   # Basic statistical functions
- tseries    # Time series analysis
- moments    # Skewness and kurtosis calculation
- VIM        # Missing data handling
- gridExtra  # Layout for graphical outputs
- lmtest     # Jarque-Bera test
- nortest    # Normality tests
- MASS       # Ordinal logistic regression
- car        # Model diagnostics
- olsrr      # Model diagnostics
- pscl       # Pseudo R-squared
- pwr        # Statistical power analysis
- dplyr      # Data manipulation
- caret      # Machine Learning
- pROC       # ROC curves
- caret      # For data preprocessing
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

1. Data Preprocessing (R/preprocessing.R)
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

3. Normality Tests (R/statistical_tests.R)
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
  - Ordinal logistic regression
  - VIF analysis for multicollinearity
  - Model evaluation metrics

8. ROC Analysis (R/modeling.R)
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
  title = {CardioSTAT: A Comprehensive Statistical Framework for Heart Disease Analysis},
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
