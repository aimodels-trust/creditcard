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

# Define expected columns
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

# User input options
option = st.radio("Select Input Method", ["Single Entry", "Batch Upload (CSV)"])

if option == "Single Entry":
    # Single input fields
    limit_bal = st.number_input("Credit Limit", min_value=0)
    age = st.number_input("Age", min_value=18)
    sex = st.selectbox("Gender", list(sex_mapping.keys()))
    education = st.selectbox("Education", list(education_mapping.keys()))
    marriage = st.selectbox("Marital Status", list(marriage_mapping.keys()))
    pay_0 = st.selectbox("Repayment Status", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    bill_amt1 = st.number_input("Bill Amount", min_value=0)
    pay_amt1 = st.number_input("Payment Amount", min_value=0)

    # Predict button
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'LIMIT_BAL': [limit_bal],
            'AGE': [age],
            'SEX': [sex_mapping[sex]],
            'EDUCATION': [education_mapping[education]],
            'MARRIAGE': [marriage_mapping[marriage]],
            'PAY_0': [pay_0],
            'BILL_AMT1': [bill_amt1],
            'PAY_AMT1': [pay_amt1]
        })

        # Add missing columns with default values
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match the model
        input_data = input_data[expected_columns]

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Display results
        if prediction == 1:
            st.error(f"Default Risk: High ({probability:.2%})")
        else:
            st.success(f"Default Risk: Low ({probability:.2%})")

elif option == "Batch Upload (CSV)":
    st.write("**Expected CSV format:**")
    st.write(pd.DataFrame(columns=expected_columns).head())
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Ensure all expected columns exist in the file
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Handle categorical variables safely
        df['SEX'] = df['SEX'].map(sex_mapping).fillna(0).astype(int)
        df['EDUCATION'] = df['EDUCATION'].map(education_mapping).fillna(0).astype(int)
        df['MARRIAGE'] = df['MARRIAGE'].map(marriage_mapping).fillna(0).astype(int)
        
        df = df[expected_columns]

        # Make batch predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        df['Default_Risk'] = predictions
        df['Probability'] = probabilities

        st.write(df[['Default_Risk', 'Probability']])
        st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
