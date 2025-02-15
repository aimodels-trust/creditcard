import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import gdown
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Google Drive file ID
file_id = "1a5xUtoaiutoXY0EMfL0vIL3A6DWJSyUs"
url = f"https://drive.google.com/uc?id={file_id}"

# Download the model
output = "credit_default_model.pkl"
gdown.download(url, output, quiet=False)

# Load Model and Scaler
model = joblib.load("credit_default_model.pkl")
scaler = joblib.load("scaler.pkl")  # Ensure scaler.pkl is in your working directory

# Streamlit UI
st.set_page_config(page_title="Credit Default Prediction", layout="wide")
st.title("💳 Credit Default Prediction App")
st.markdown("---")

st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head())
    
    # Ensure column names match training data
    required_columns = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", 
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]
    
    if all(col in df.columns for col in required_columns):
        # Preprocess Data
        X = df[required_columns]
        X_scaled = scaler.transform(X)
        
        # Make Predictions
        predictions = model.predict(X_scaled)
        df["Default Prediction"] = predictions
        
        # Display Predictions
        st.write("### Predictions:")
        st.dataframe(df[["Default Prediction"]])
        
        # SHAP Explainability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled[:50])  # Limit to 50 samples for performance
        
        st.write("### Explainability (SHAP Visualization)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig)
        
    else:
        st.error("Uploaded file does not match the required columns! Please check the dataset.")
else:
    st.warning("Please upload a CSV file.")

st.markdown("---")
st.write("💡 Developed with ❤️ using Streamlit")
