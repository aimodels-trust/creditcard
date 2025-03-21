import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("credit_default_model.pkl")

# Define categorical mappings
sex_mapping = {"Male": 1, "Female": 2}
education_mapping = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
marriage_mapping = {"Married": 1, "Single": 2, "Others": 3}

# Streamlit App
st.title("Credit Card Default Prediction")

# Input fields
limit_bal = st.number_input("Credit Limit", min_value=0)
age = st.number_input("Age", min_value=18)
sex = st.selectbox("Gender", list(sex_mapping.keys()))
education = st.selectbox("Education", list(education_mapping.keys()))
marriage = st.selectbox("Marital Status", list(marriage_mapping.keys()))
pay_0 = st.selectbox("Repayment Status (Most Recent Month)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Inputs for additional months
bill_amt1 = st.number_input("Bill Amount (Most Recent Month)", min_value=0)
bill_amt2 = st.number_input("Bill Amount (2nd Most Recent Month)", min_value=0)
bill_amt3 = st.number_input("Bill Amount (3rd Most Recent Month)", min_value=0)
bill_amt4 = st.number_input("Bill Amount (4th Most Recent Month)", min_value=0)
bill_amt5 = st.number_input("Bill Amount (5th Most Recent Month)", min_value=0)
bill_amt6 = st.number_input("Bill Amount (6th Most Recent Month)", min_value=0)

pay_amt1 = st.number_input("Payment Amount (Most Recent Month)", min_value=0)
pay_amt2 = st.number_input("Payment Amount (2nd Most Recent Month)", min_value=0)
pay_amt3 = st.number_input("Payment Amount (3rd Most Recent Month)", min_value=0)
pay_amt4 = st.number_input("Payment Amount (4th Most Recent Month)", min_value=0)
pay_amt5 = st.number_input("Payment Amount (5th Most Recent Month)", min_value=0)
pay_amt6 = st.number_input("Payment Amount (6th Most Recent Month)", min_value=0)

# Inputs for PAY_2 to PAY_6
pay_2 = st.selectbox("Repayment Status (2nd Most Recent Month)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
pay_3 = st.selectbox("Repayment Status (3rd Most Recent Month)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
pay_4 = st.selectbox("Repayment Status (4th Most Recent Month)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
pay_5 = st.selectbox("Repayment Status (5th Most Recent Month)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
pay_6 = st.selectbox("Repayment Status (6th Most Recent Month)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'LIMIT_BAL': [limit_bal],
        'AGE': [age],
        'SEX': [sex_mapping[sex]],
        'EDUCATION': [education_mapping[education]],
        'MARRIAGE': [marriage_mapping[marriage]],
        'PAY_0': [pay_0],
        'PAY_2': [pay_2],
        'PAY_3': [pay_3],
        'PAY_4': [pay_4],
        'PAY_5': [pay_5],
        'PAY_6': [pay_6],
        'BILL_AMT1': [bill_amt1],
        'BILL_AMT2': [bill_amt2],
        'BILL_AMT3': [bill_amt3],
        'BILL_AMT4': [bill_amt4],
        'BILL_AMT5': [bill_amt5],
        'BILL_AMT6': [bill_amt6],
        'PAY_AMT1': [pay_amt1],
        'PAY_AMT2': [pay_amt2],
        'PAY_AMT3': [pay_amt3],
        'PAY_AMT4': [pay_amt4],
        'PAY_AMT5': [pay_amt5],
        'PAY_AMT6': [pay_amt6],
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display results
    if prediction == 1:
        st.error(f"Default Risk: High ({probability:.2%})")
    else:
        st.success(f"Default Risk: Low ({probability:.2%})")
