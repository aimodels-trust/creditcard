import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np

# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    st.write("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load the trained model
model = joblib.load(model_path)

# Step 3: Define the app
st.title("Credit Card Default Prediction")

# Option for Single or Batch Prediction
prediction_type = st.radio("Choose Prediction Type", ["Single Prediction", "Batch Prediction"])

# Define input fields for single prediction
if prediction_type == "Single Prediction":
    limit_bal = st.number_input("Credit Limit", min_value=0)
    age = st.number_input("Age", min_value=18)
    sex = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
    marriage = st.selectbox("Marital Status", ["Married", "Single", "Others"])
    pay_0 = st.selectbox("Repayment Status", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    bill_amt1 = st.number_input("Bill Amount", min_value=0)
    pay_amt1 = st.number_input("Payment Amount", min_value=0)
    
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'LIMIT_BAL': [limit_bal], 'AGE': [age], 'SEX': [sex], 'EDUCATION': [education],
            'MARRIAGE': [marriage], 'PAY_0': [pay_0], 'BILL_AMT1': [bill_amt1], 'PAY_AMT1': [pay_amt1]
        })

        # Mapping categorical variables
        mappings = {
            'SEX': {'Male': 1, 'Female': 2},
            'EDUCATION': {'Graduate School': 1, 'University': 2, 'High School': 3, 'Others': 4},
            'MARRIAGE': {'Married': 1, 'Single': 2, 'Others': 3}
        }
        
        for col, mapping in mappings.items():
            input_data[col] = input_data[col].map(mapping)
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.error(f"Default Risk: High ({probability:.2%})")
        else:
            st.success(f"Default Risk: Low ({probability:.2%})")

# Batch Prediction
elif prediction_type == "Batch Prediction":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df.head())
        
        # Map categorical values
        mappings = {
            'SEX': {'Male': 1, 'Female': 2},
            'EDUCATION': {'Graduate School': 1, 'University': 2, 'High School': 3, 'Others': 4},
            'MARRIAGE': {'Married': 1, 'Single': 2, 'Others': 3}
        }
        
        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        df['Prediction'] = predictions
        df['Probability'] = probabilities
        df['Risk Level'] = df['Prediction'].apply(lambda x: "High" if x == 1 else "Low")
        
        st.write("Predictions:")
        st.dataframe(df[['Prediction', 'Probability', 'Risk Level']])
        
        # Allow downloading results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", data=csv, file_name="predictions.csv", mime="text/csv")
