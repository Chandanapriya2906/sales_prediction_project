import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ===========================
# LOAD DATASET
# ===========================
data = pd.read_csv("Advertising.csv", sep="\t")
data = data.drop("Unnamed: 0", axis=1)

# ===========================
# LOAD TRAINED MODEL
# ===========================
model = joblib.load("sales_prediction_model.pkl")

# ===========================
# SIDEBAR
# ===========================
st.sidebar.title("📋 Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "🏠 Home",
        "📁 Dataset",
        "📈 Visualizations",
        "🤖 Prediction",
        "ℹ️ About"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":

    st.title("📊 Sales Prediction Dashboard")

    st.markdown("## Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📁 Total Records", len(data))

    with col2:
        st.metric("📈 Average Sales", round(data["Sales"].mean(),2))

    with col3:
        st.metric("🏆 Highest Sales", round(data["Sales"].max(),2))

    st.markdown("---")

    st.subheader("📌 Project Description")

    st.write("""
This project predicts product sales using a **Machine Learning Linear Regression Model**.

### Features

- 📁 View Dataset
- 📈 Data Visualizations
- 🤖 Predict Sales
- 📊 Model Performance
- ℹ️ About Project
""")

    st.success("✅ Model Accuracy (R² Score): 89.94%")

    st.info("Use the left sidebar to navigate through the project.")

# =====================================================
# DATASET PAGE
# =====================================================
elif page == "📁 Dataset":

    st.title("📁 Advertising Dataset")

    st.subheader("Dataset Preview")
    st.dataframe(data)

    st.subheader("Dataset Shape")
    st.write(data.shape)

    st.subheader("Column Names")
    st.write(list(data.columns))

    st.subheader("Statistical Summary")
    st.dataframe(data.describe())

# =====================================================
# VISUALIZATION PAGE
# =====================================================
elif page == "📈 Visualizations":

    st.title("📈 Data Visualizations")

    st.subheader("TV Advertising vs Sales")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(data["TV"], data["Sales"])
    ax.set_xlabel("TV Budget")
    ax.set_ylabel("Sales")
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Radio Advertising vs Sales")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(data["Radio"], data["Sales"])
    ax.set_xlabel("Radio Budget")
    ax.set_ylabel("Sales")
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Newspaper Advertising vs Sales")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(data["Newspaper"], data["Sales"])
    ax.set_xlabel("Newspaper Budget")
    ax.set_ylabel("Sales")
    ax.grid(True)

    st.pyplot(fig)
    # ===========================
    # Sales Distribution
    # ===========================
    st.subheader("Sales Distribution")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(data["Sales"], bins=10)
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")
    ax.grid(True)

    st.pyplot(fig)

# =====================================================
# PREDICTION PAGE
# =====================================================
elif page == "🤖 Prediction":

    st.title("🤖 Sales Prediction")

    st.write("Enter the advertising budgets below.")

    col1, col2 = st.columns(2)

    with col1:
        tv = st.number_input(
            "📺 TV Advertising Budget",
            min_value=0.0,
            value=100.0
        )

        radio = st.number_input(
            "📻 Radio Advertising Budget",
            min_value=0.0,
            value=20.0
        )

    with col2:
        newspaper = st.number_input(
            "📰 Newspaper Advertising Budget",
            min_value=0.0,
            value=30.0
        )

    st.markdown("---")

    if st.button("🚀 Predict Sales"):

        input_data = pd.DataFrame({
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        })

        prediction = model.predict(input_data)

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="📈 Predicted Sales",
            value=f"{prediction[0]:.2f} Units"
        )

        st.balloons()

# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
# Sales Prediction Using Machine Learning

## Objective

Predict future product sales using advertising budgets.

---

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

---

## Machine Learning Algorithm

Linear Regression

---

## Dataset Features

- TV Advertising
- Radio Advertising
- Newspaper Advertising

Target Variable:

- Sales

---

## Model Performance

- MAE : 1.4608
- MSE : 3.1741
- RMSE : 1.7816
- R² Score : 89.94%

---

## Developed By

Sales Prediction Dashboard using Machine Learning.
""")