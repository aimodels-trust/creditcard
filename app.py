import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1tdfigkbM6cfk7-Emyd2jueEcS2vsb5oP"
model_path = "credit_default_model.pkl"

# Download model if not present
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load the trained model
model = joblib.load(model_path)

# Define the app
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
    input_data = pd.DataFrame({
        'LIMIT_BAL': [limit_bal],
        'AGE': [age],
        'SEX': [sex],
        'EDUCATION': [education],
        'MARRIAGE': [marriage],
        'PAY_0': [pay_0],
        'BILL_AMT1': [bill_amt1],
        'PAY_AMT1': [pay_amt1]
    })

    # Expected columns based on training
    expected_columns = [
        'LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

    # Mapping categorical variables
    sex_mapping = {'Male': 1, 'Female': 2}
    education_mapping = {'Graduate School': 1, 'University': 2, 'High School': 3, 'Others': 4}
    marriage_mapping = {'Married': 1, 'Single': 2, 'Others': 3}

    input_data['SEX'] = input_data['SEX'].map(sex_mapping).fillna(1).astype(int)
    input_data['EDUCATION'] = input_data['EDUCATION'].map(education_mapping).fillna(2).astype(int)
    input_data['MARRIAGE'] = input_data['MARRIAGE'].map(marriage_mapping).fillna(1).astype(int)

    # Add missing columns with default values
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Default value for missing columns

    # Convert all values to integer
    input_data = input_data.astype(int)

    # Ensure the column order is correct
    input_data = input_data[expected_columns]

    # Debugging output
    st.write("Model expects features:", model.feature_names_in_)
    st.write("Input Data Preview:", input_data)

    # Make prediction
    try:
        probability = model.predict_proba(input_data)[0][1]
        scaled_probability = min(probability * 2, 1.0)

        if scaled_probability >= 0.5:
            st.error(f"Default Risk: High ({scaled_probability:.2%})")
        else:
            st.success(f"Default Risk: Low ({scaled_probability:.2%})")

    except ValueError as e:
        st.error(f"Prediction error: {e}")
        st.write("Unique Values in Each Column:", {col: input_data[col].unique() for col in input_data.columns})
