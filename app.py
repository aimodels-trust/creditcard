import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Set Streamlit Page Configuration
st.set_page_config(page_title="Credit Default Prediction", layout="wide")

# Download Model from Google Drive (if not available)
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Streamlit Title
st.title("💳 Credit Card Default Prediction with Explainability")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["🏠 Home", "📊 Feature Importance"])

# Define expected input features
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

if app_mode == "🏠 Home":
    st.write("### Predict Credit Card Default")

    # Manual input form
    with st.form("user_input_form"):
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0)
        age = st.number_input("Age (AGE)", min_value=18, max_value=100)
        sex = st.selectbox("Sex (SEX)", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Others"}[x])
        marriage = st.selectbox("Marriage (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
        
        # Repayment Status
        pay_0 = st.number_input("Repayment Status (PAY_0)", min_value=-2, max_value=8)
        pay_2 = st.number_input("Repayment Status (PAY_2)", min_value=-2, max_value=8)
        pay_3 = st.number_input("Repayment Status (PAY_3)", min_value=-2, max_value=8)
        pay_4 = st.number_input("Repayment Status (PAY_4)", min_value=-2, max_value=8)
        pay_5 = st.number_input("Repayment Status (PAY_5)", min_value=-2, max_value=8)
        pay_6 = st.number_input("Repayment Status (PAY_6)", min_value=-2, max_value=8)

        # Bill Amounts
        bill_amt1 = st.number_input("Bill Amount 1 (BILL_AMT1)", min_value=0)
        bill_amt2 = st.number_input("Bill Amount 2 (BILL_AMT2)", min_value=0)
        bill_amt3 = st.number_input("Bill Amount 3 (BILL_AMT3)", min_value=0)
        bill_amt4 = st.number_input("Bill Amount 4 (BILL_AMT4)", min_value=0)
        bill_amt5 = st.number_input("Bill Amount 5 (BILL_AMT5)", min_value=0)
        bill_amt6 = st.number_input("Bill Amount 6 (BILL_AMT6)", min_value=0)

        # Payment Amounts
        pay_amt1 = st.number_input("Payment Amount 1 (PAY_AMT1)", min_value=0)
        pay_amt2 = st.number_input("Payment Amount 2 (PAY_AMT2)", min_value=0)
        pay_amt3 = st.number_input("Payment Amount 3 (PAY_AMT3)", min_value=0)
        pay_amt4 = st.number_input("Payment Amount 4 (PAY_AMT4)", min_value=0)
        pay_amt5 = st.number_input("Payment Amount 5 (PAY_AMT5)", min_value=0)
        pay_amt6 = st.number_input("Payment Amount 6 (PAY_AMT6)", min_value=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare user input data (Fixing Shape Issue)
        user_inputs = {
            "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
            "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
            "BILL_AMT1": bill_amt1, "BILL_AMT2": bill_amt2, "BILL_AMT3": bill_amt3, "BILL_AMT4": bill_amt4, "BILL_AMT5": bill_amt5, "BILL_AMT6": bill_amt6,
            "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3, "PAY_AMT4": pay_amt4, "PAY_AMT5": pay_amt5, "PAY_AMT6": pay_amt6
        }

        user_data = pd.DataFrame([user_inputs])  # Wrap in a list to ensure 2D format

        # Make prediction
        prediction = model.predict(user_data)
        probability = model.predict_proba(user_data)[:, 1]

        st.write("### Prediction Result")
        st.write(f"Default Risk: {'High' if prediction[0] == 1 else 'Low'}")
        st.write(f"Probability of Default: {probability[0]:.2f}")

elif app_mode == "📊 Feature Importance":
    st.write("### 🔍 Feature Importance & Explainability")

    uploaded_file = st.file_uploader("📂 Upload CSV for SHAP Analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV format is incorrect! Check the column count.")
        else:
            X_transformed = df  # No need for preprocessing

            # SHAP Explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)

            if isinstance(shap_values, list):  # Binary classification case
                shap_values = shap_values[1]  

            # Feature Importance
            shap_importance = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({'Feature': expected_columns, 'SHAP Importance': shap_importance})
            importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).head(10)

            st.write("### 🔥 Top 10 Most Important Features")
            st.dataframe(importance_df)

            # Plot Feature Importance
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="royalblue")
            ax.set_xlabel("SHAP Importance")
            ax.set_ylabel("Feature")
            ax.set_title("📊 Feature Importance")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            # SHAP Summary Plot
            st.write("### 📊 SHAP Summary Plot")
            shap.summary_plot(shap_values, X_transformed, feature_names=expected_columns, show=False)
            plt.savefig("shap_summary.png", bbox_inches='tight')
            st.image("shap_summary.png")
