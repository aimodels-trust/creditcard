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

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load the trained model
model = joblib.load(model_path)

# Step 3: Define Streamlit app
st.title("Credit Card Default Prediction with Explainability")

# Define expected input features
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

    if df.shape[1] != len(expected_columns):
        st.error("Uploaded CSV does not match expected format. Please check the number of columns.")
    else:
        # Extract preprocessor and classifier
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Transform input data
        X_transformed = preprocessor.transform(df)

        # Get correct transformed feature names
        feature_names = preprocessor.get_feature_names_out()

        # Make predictions
        predictions = classifier.predict(X_transformed)
        probabilities = classifier.predict_proba(X_transformed)[:, 1]

        df['Default_Risk'] = predictions
        df['Probability'] = probabilities

        st.write("### Prediction Results")
        st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

        # SHAP Explainability
        st.write("### Feature Importance")

        # Use a small sample (to speed up SHAP)
        sample_data = X_transformed[:50]

        # Use SHAP TreeExplainer (since it's a RandomForest model)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(sample_data)

        # SHAP summary plot with correct feature names
        shap.summary_plot(shap_values[1], sample_data, feature_names=feature_names, show=False)
        plt.savefig("shap_summary.png", bbox_inches='tight')
        st.image("shap_summary.png")
