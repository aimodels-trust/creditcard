import streamlit as st
import pandas as pd
import joblib
import requests
import os

# GitHub raw file link (replace with your actual link)
model_url = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/main/model.pkl"
model_path = "model.pkl"

# Check if the model file exists; if not, download it
if not os.path.exists(model_path):
    st.info("Downloading model from GitHub...")
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            f.write(response.content)
    else:
        st.error("Failed to download the model. Check the GitHub link.")
        st.stop()

# Load the trained model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit App UI
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

    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Force 100% probability for extreme high-risk cases
        if pay_0 >= 8 and bill_amt1 > 40000 and pay_amt1 == 0:
            probability = 1.0
            prediction = 1

        # Display results
        if prediction == 1:
            st.error(f"Default Risk: High ({probability:.2%})")
        else:
            st.success(f"Default Risk: Low ({probability:.2%})")

    except Exception as e:
        st.error(f"Prediction error: {e}")
