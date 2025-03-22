import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model.pkl"

# Check if the model file already exists; if not, download it
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load the trained model
pipeline = joblib.load(model_path)
model = pipeline.named_steps["classifier"]  # Extract the classifier from the pipeline
preprocessor = pipeline.named_steps["preprocessor"]  # Extract the preprocessor

# Streamlit UI
st.title("Credit Card Default Prediction with Explainability")

# Define expected columns (excluding the target variable)
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

st.write("## Batch Upload (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = expected_columns

    # Transform the data using the pipeline's preprocessor
    transformed_data = preprocessor.transform(df)

    # Make predictions
    predictions = model.predict(transformed_data)
    probabilities = model.predict_proba(transformed_data)[:, 1]

    df['Default_Risk'] = predictions
    df['Probability'] = probabilities

    st.write("### Prediction Results")
    st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

    st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

    # Explainability using SHAP (use only a subset for performance)
    sample_data = transformed_data[:500]  # Limit SHAP computation to 500 rows
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_data)

    # Display Feature Importance
    st.write("### Feature Importance")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_data, show=False)
    plt.savefig("shap_summary.png", bbox_inches='tight')
    st.image("shap_summary.png")
