import streamlit as st
import pandas as pd
import joblib
import shap
import gdown
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Download model from Google Drive
model_path = "credit_default_model.pkl"
gdrive_url = "https://drive.google.com/uc?id=1tdfigkbM6cfk7-Emyd2jueEcS2vsb5oP"

if not os.path.exists(model_path):
    gdown.download(gdrive_url, model_path, quiet=False)

# Load the trained model
model = joblib.load(model_path)

# Streamlit App
st.title("Credit Card Default Prediction")

# Input fields
limit_bal = st.number_input("Credit Limit", min_value=0)
age = st.number_input("Age", min_value=18)
sex = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
marriage = st.selectbox("Marital Status", ["Married", "Single", "Others"])
pay_0 = st.selectbox("Repayment Status", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
bill_amt1 = st.number_input("Bill Amount", min_value=0)
pay_amt1 = st.number_input("Payment Amount", min_value=0)

# Mapping categorical values
sex_map = {"Male": 1, "Female": 2}
edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
marriage_map = {"Married": 1, "Single": 2, "Others": 3}

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'LIMIT_BAL': [limit_bal],
        'AGE': [age],
        'SEX': [sex_map[sex]],
        'EDUCATION': [edu_map[education]],
        'MARRIAGE': [marriage_map[marriage]],
        'PAY_0': [pay_0],
        'BILL_AMT1': [bill_amt1],
        'PAY_AMT1': [pay_amt1]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Display results
    if prediction == 1:
        st.error(f"Default Risk: High ({probability:.2%})")
    else:
        st.success(f"Default Risk: Low ({probability:.2%})")
    
    # SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    
    st.subheader("Feature Contributions")
    st.write("The following chart explains the contribution of each feature to the prediction:")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0])
    st.pyplot(fig)
