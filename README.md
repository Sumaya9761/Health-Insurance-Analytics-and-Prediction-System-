# Health Insurance Analytics & Prediction

> An end-to-end regression pipeline predicting the number of insured residents (NIC) from US county-level insurance demographics, improved through dimensionality reduction and hyperparameter tuning to a test R² of 0.968.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)

## Overview

This project predicts the number of insured residents from US insurance demographic data (2022), spanning geography, age, race, sex, and income categories. Six regression models are compared, the strongest is selected, and its performance is then improved through dimensionality reduction and hyperparameter tuning. Clustering and anomaly-detection are also evaluated as feature-engineering experiments.

## Key Results

Models were evaluated on a held-out 30% test set (R² reported on test).

| Model | Test R² |
|---|---|
| Linear Regression | 0.693 |
| Ridge Regression | 0.690 |
| Lasso Regression | 0.694 |
| Gradient Boosting | 0.712 |
| Decision Tree | 0.835 |
| Random Forest | 0.859 |
| Random Forest + PCA | 0.911 |
| **Random Forest + PCA + tuning** | **0.968** |

**Final model:** Random Forest on 10 PCA components, tuned with GridSearchCV (144 candidates × 3-fold CV), reaching **test R² = 0.968** (RMSE ≈ 41.3k, MAE ≈ 8.4k).

## Approach

- **Preprocessing:** dropped margin-of-error columns, coerced numeric types, handled missing values, scaled numerical features (StandardScaler) and one-hot encoded categoricals (geography, age, race, sex, income)
- **EDA:** correlation heatmaps, distribution plots, and demographic breakdowns of insured/uninsured rates by state, age, race, and income
- **Modeling:** compared 6 regressors (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting)
- **Improvement on the best model:** PCA (10 components) for dimensionality reduction, then GridSearchCV hyperparameter tuning
- **Additional experiments:** K-Means and DBSCAN cluster labels as added features, and Isolation Forest anomaly removal — evaluated but did not outperform RF + PCA



## License

Released under the MIT License — see [LICENSE](LICENSE).
