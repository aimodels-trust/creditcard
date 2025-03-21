import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np

# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model.pkl"

# Check if the model file already exists; if not, download it
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load the trained model
model = joblib.load(model_path)

# Step 3: Define the app
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

# Predict button
if st.button("Predict"):
    # Map string labels to numeric values
    sex_mapping = {'Male': 1, 'Female': 2}
    education_mapping = {'Graduate School': 1, 'University': 2, 'High School': 3, 'Others': 4}
    marriage_mapping = {'Married': 1, 'Single': 2, 'Others': 3}

    # Prepare input data
    input_data = pd.DataFrame({
        'LIMIT_BAL': [limit_bal],
        'AGE': [age],
        'SEX': [sex_mapping[sex]],
        'EDUCATION': [education_mapping[education]],
        'MARRIAGE': [marriage_mapping[marriage]],
        'PAY_0': [pay_0], 'PAY_2': [pay_0], 'PAY_3': [pay_0], 'PAY_4': [pay_0], 'PAY_5': [pay_0], 'PAY_6': [pay_0],
        'BILL_AMT1': [bill_amt1], 'BILL_AMT2': [bill_amt1], 'BILL_AMT3': [bill_amt1],
        'BILL_AMT4': [bill_amt1], 'BILL_AMT5': [bill_amt1], 'BILL_AMT6': [bill_amt1],
        'PAY_AMT1': [pay_amt1], 'PAY_AMT2': [pay_amt1], 'PAY_AMT3': [pay_amt1],
        'PAY_AMT4': [pay_amt1], 'PAY_AMT5': [pay_amt1], 'PAY_AMT6': [pay_amt1]
    })

    # Ensure input data matches model expectations
    expected_columns = [
        'LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    input_data = input_data[expected_columns]

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display results
    if prediction == 1:
        st.error(f"Default Risk: High ({probability:.2%})")
    else:
        st.success(f"Default Risk: Low ({probability:.2%})")
