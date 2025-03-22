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
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
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
    input_values = {}
    for col in expected_columns:
        if col == 'SEX':
            input_values[col] = sex_mapping[st.selectbox("Gender", list(sex_mapping.keys()))]
        elif col == 'EDUCATION':
            input_values[col] = education_mapping[st.selectbox("Education", list(education_mapping.keys()))]
        elif col == 'MARRIAGE':
            input_values[col] = marriage_mapping[st.selectbox("Marital Status", list(marriage_mapping.keys()))]
        elif col.startswith("PAY_AMT"):
            input_values[col] = st.number_input(f"{col}", min_value=0)
        elif col.startswith("PAY_"):
            input_values[col] = st.number_input(f"{col}", min_value=-2, max_value=9, step=1)
        else:
            input_values[col] = st.number_input(col, min_value=0)
    
    # Predict button
    if st.button("Predict"):
        input_data = pd.DataFrame([input_values])
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
        df = pd.read_csv(uploaded_file, header=None)
        df.columns = expected_columns
        
        # Make batch predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        df['Default_Risk'] = predictions
        df['Probability'] = probabilities
        
        st.write(df[['Default_Risk', 'Probability']])
        st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
