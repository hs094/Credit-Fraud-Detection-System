# Credit Card Fraud Detection Project

A machine learning project that implements multiple models to detect fraudulent credit card transactions with high precision.

## Overview

This project focuses on detecting credit card fraud using various machine learning algorithms while dealing with highly imbalanced datasets. The implementation achieves high accuracy in identifying fraudulent transactions while minimizing false positives.

## Key Features

- Handles highly imbalanced dataset (only 0.172% fraud cases)
- Implements multiple ML models:
  - Random Forest Classifier (AUC: 0.85)
  - AdaBoost Classifier (AUC: 0.83)
  - CatBoost Classifier (AUC: 0.86)
  - XGBoost (AUC: 0.974)
  - LightGBM (AUC: 0.946)
- Extensive data exploration and visualization
- Feature importance analysis
- Cross-validation implementation

## Dataset

The dataset contains credit card transactions made by European cardholders in September 2013:

- Total transactions: 284,807
- Fraudulent cases: 492 (0.172%)
- Features: 31 (Time, V1-V28, Amount, Class)
- All features are numerical and PCA transformed except Time and Amount

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
catboost
plotly
```

## Project Structure

```
Credit Fraud Detection Project/
├── credit-card-fraud-detection-predictive-models.ipynb
├── credit-fraud-dealing-with-imbalanced-datasets.ipynb
├── requirements.txt
├── input/
│   └── creditcard.csv
└── catboost_info/
    ├── catboost_training.json
    ├── learn_error.tsv
    └── time_left.tsv
```

## Models Performance

| Model | AUC Score |
|-------|-----------|
| XGBoost | 0.974 |
| LightGBM | 0.946 |
| CatBoost | 0.86 |
| Random Forest | 0.85 |
| AdaBoost | 0.83 |

## Key Findings

- Feature importance analysis revealed V17, V12, V14, V16, V11, V10 as the most significant predictors
- XGBoost achieved the best performance with 0.974 AUC score
- Cross-validation helped ensure model stability and generalization
- The models successfully handle the extreme class imbalance

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebooks:
```bash
jupyter notebook
```

## Usage

1. Open [`credit-card-fraud-detection-predictive-models.ipynb`](credit-card-fraud-detection-predictive-models.ipynb) for the main analysis and model implementations
2. Open [`credit-fraud-dealing-with-imbalanced-datasets.ipynb`](credit-fraud-dealing-with-imbalanced-datasets.ipynb) for handling imbalanced dataset techniques

## License

This project uses anonymized credit card transaction data and is intended for educational purposes only.