# 🚗 Used Car Price Prediction

An end-to-end Machine Learning project for predicting used car prices based on vehicle specifications. The project covers the complete data science workflow, from data cleaning and exploratory data analysis to feature engineering, hypothesis testing and model development.

---

## 📌 Project Overview

The objective of this project is to build a machine learning model capable of estimating the price of a used car using its specifications such as:

- Brand
- Mileage
- Fuel Type
- Transmission
- Accident History
- Horsepower
- Engine Size
- Number of Cylinders
- Vehicle Age

---

## 📂 Project Structure

```
Used-Car-Price-Prediction/
│
├── data/
│   ├── raw_data.csv
│   └── clean_data.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   └── 04_machine_learning.ipynb
│
├── models/
│   └── car_price_model.pkl
│
│
├── requirements.txt
└── README.md
```

---

## 🛠 Technologies

- Python
- Pandas
- NumPy
- Scipy
- Matplotlib
- Scikit-learn
- Joblib

---

## 📊 Data Cleaning

Performed several preprocessing steps including:

- Handling missing values
- Removing duplicates
- Standardizing categorical values
- Extracting:
  - Horsepower (HP)
  - Engine Size
  - Number of Cylinders
- Creating a new **Age** feature
- Preparing a clean dataset for modeling

---

## 📈 Exploratory Data Analysis

Performed both univariate and bivariate analysis.

Some key findings include:

- Vehicle price decreases as mileage increases.
- Higher horsepower generally leads to higher prices.
- Accident history significantly affects vehicle prices.
- Gasoline is the dominant fuel type.
- Automatic transmission is the most common transmission type.

---

## 📑 Hypothesis Testing

Statistical tests were performed to validate insights.

- Independent Two-Sample t-test
- One-Way ANOVA
- Pearson Correlation

---

## ⚙️ Feature Engineering

Applied several feature engineering techniques:

- Log Transformation of the target variable
- Feature Extraction
- Age Calculation
- Missing Value Imputation
- One-Hot Encoding
- Standard Scaling
- ColumnTransformer
- Scikit-learn Pipeline

---

## 🤖 Machine Learning Models

The following models were evaluated:

| Model | Test R² | Cross Validation |
|--------|---------:|----------------:|
| Linear Regression | 76.7% | 82.0% |
| Polynomial Regression | 79.4% | 83.4% |
| Decision Tree | 74.3% | 76.1% |
| **Random Forest** ⭐ | **82.2%** | **86.0%** |

Random Forest achieved the best overall performance and was selected as the final model.

---

## 🚀 Future Improvements

- Experiment with XGBoost and CatBoost.
- Add advanced feature engineering.
- Improve prediction performance for luxury vehicles.
- Deploy the model using Streamlit.

---

## 👨‍💻 Author

**Mina Ekramy**

Computer Engineering Student  
Cairo University

GitHub: https://github.com/minaekramy12

LinkedIn: https://linkedin.com/in/minaekramy