# sales_prediction_project
A machine learning project that predicts product sales based on advertising budgets using Linear Regression. The project includes data preprocessing, model training, performance evaluation, and an interactive Streamlit dashboard for visualization and sales prediction.
# 📈 Sales Prediction using Machine Learning

A Data Science and Machine Learning project focused on forecasting future sales based on historical data, store features, promotional activities, and seasonal trends.

---

## 📌 Project Overview

Accurate sales forecasting helps businesses optimize inventory management, allocate resources effectively, and maximize revenue. This project aims to build an end-to-end predictive model that forecasts future sales performance using historical transaction data and external features.

### Key Objectives
- Analyze sales patterns, seasonality, and feature impact.
- Handle missing values, outliers, and data preprocessing.
- Engineer domain-specific features (lag features, rolling averages, temporal trends).
- Train and evaluate multiple Machine Learning models to find the best performer.
- Deploy / save the trained model for downstream inference.

---

## 📊 Dataset Overview

The dataset includes transaction records along with metadata about stores, promotions, and time features:

- **Target Variable:** `Sales` — Total monetary sales for a given date/store.
- **Features:**
  - `Store_ID`: Unique identifier for each store.
  - `Date`: Transaction date.
  - `Promotions`: Binary/numeric indicator of promotional campaigns.
  - `Customers`: Number of customers visiting on a given day.
  - `Holiday_Flag`: Indicator for holidays/special events.
  - `Store_Type / Category`: Classification of the retail outlet.

---

## 🛠️ Project Architecture & Workflow

```text
├── data/
│   ├── raw/             # Original, untouched data
│   └── processed/       # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Preprocessing & Feature Creation
│   └── 03_model_training.ipynb      # Training & Evaluation
├── models/              # Saved model artifacts (.pkl, .joblib)
├── src/
│   ├── preprocess.py    # Data cleaning functions
│   └── train.py         # Model training pipeline
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
