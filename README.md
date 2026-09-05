# 🚗 Used Car Price Prediction & Interactive Web App

<img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTs2NHHCAHw0gl8eIkoHcIrbndZOGKP5nQzNQWGNMbaABBkqjM0TfV8AvA&s=10' width=1000 />


An end-to-end Machine Learning project for predicting used car prices using an optimized **XGBoost Pipeline** and an interactive **Streamlit** web application. This repository covers the complete data science workflow, including data cleaning, exploratory data analysis, hypothesis testing, pipeline feature engineering, model optimization, and deployment.

---

## 📌 Project Overview

The objective of this project is to estimate the market price of a used car based on key physical, mechanical, and historical attributes:

- **Categorical Features:** `brand`, `fuel_type`, `transmission`, `accident`
- **Numerical Features:** `milage`, `HP`, `engine_size`, `n_cylinders`, `age`

The target variable (`price`) is modeled using a log transformation (`np.log1p`) during training to handle skewness and reversed (`np.expm1`) during inference.

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
│   └── xgboost_pipeline.joblib
│
├── app.py
├── requirements.txt
└── README.md

```


---


## 🛠 Technologies & Tools

- **Programming:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy, SciPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** XGBoost, Scikit-learn
- **Model Deployment & Web Interface:** Streamlit
- **Model Serialization:** Joblib

---

## 📊 Data Cleaning & Preprocessing

- Handled missing categorical values (e.g., imputing missing `fuel_type` values with `'Unknown'`).
- Removed duplicated records and formatted noisy text attributes.
- Parsed and extracted clean numerical values for `HP`, `engine_size`, and `n_cylinders`.
- Derived the `age` feature directly from vehicle manufacturing years (`2026 - model_year`).

---

## 📈 Exploratory Data Analysis & Hypothesis Testing

Key findings confirmed across EDA and statistical validation:

- **Mileage vs. Price:** Strong negative correlation; price depreciates non-linearly with increased mileage.
- **Engine Metrics:** `HP` and `engine_size` exhibit strong positive linear correlations with price.
- **Accident History:** Independent t-tests confirmed a statistically significant drop in market value for cars with reported accidents or damage.
- **ANOVA Testing:** Demonstrated significant variance in pricing across distinct `brand` categories and `fuel_type` variants.

---

## ⚙️ Pipeline & Feature Engineering

All preprocessing steps are encapsulated within a Scikit-Learn `Pipeline` and `ColumnTransformer` to eliminate data leakage:

- **Target Transformation:** Applied `np.log1p(y)` prior to training to normalize target distribution.
- **Categorical Encoding:** One-Hot Encoding for `brand`, `fuel_type`, `transmission`, and `accident`.
- **Numerical Scaling:** Standard scaling applied to numerical inputs where applicable.
- **Dynamic Age Calculation:** Streamlit interface automatically derives `age` from user-selected `model_year`.

---

## 🤖 Machine Learning Models & Results

Multiple algorithms were evaluated prior to hyperparameter tuning and final selection:

| Model | Test R² |
| :--- | :---: |
| Linear Regression | 76.7% |
| Decision Tree | 74.3% |
| Random Forest | 82.2% |
| **XGBoost Pipeline** ⭐ | **84.9%** |

**XGBoost** combined with the feature preprocessing pipeline achieved the highest generalization score and was selected for production deployment.

---

## 💻 Web Application (Streamlit)

An interactive interface allows users to select vehicle specifications and receive real-time price predictions.

### Running the App Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/minaekramy12/Used-Car-Price-Prediction.git
   
   cd Used-Car-Price-Prediction
   ```

2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```

3. Launch Streamlit:
   ```bash
    streamlit run app.py
   ```


---


## 👨‍💻 Author
Mina Ekramy

Computer Engineering Student, Cairo University

GitHub: [minaekramy12](https://github.com/minaekramy12/)

LinkedIn: [minaekramy](https://www.linkedin.com/in/minaekramy/)
