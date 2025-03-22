import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="Credit Default Prediction", layout="wide")

# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    st.info("Downloading model... Please wait ⏳")
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load the trained model
model = joblib.load(model_path)

# Step 3: Define Streamlit app
st.title("🔍 Credit Card Default Prediction with Explainability")

# Define expected input features
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

    if df.shape[1] != len(expected_columns):
        st.error("❌ Uploaded CSV does not match expected format. Please check the number of columns.")
    else:
        # Extract preprocessor and classifier
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Transform input data
        X_transformed = preprocessor.transform(df)

        # Get transformed feature names
        feature_names = preprocessor.get_feature_names_out()

        # Make predictions
        predictions = classifier.predict(X_transformed)
        probabilities = classifier.predict_proba(X_transformed)[:, 1]

        df['Default_Risk'] = predictions
        df['Probability'] = probabilities

        st.subheader("📊 Prediction Results")
        st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

        # SHAP Explainability
        st.subheader("📈 Feature Importance & Explainability")

        # Use a small sample (to speed up SHAP)
        sample_data = X_transformed[:50]

        # Use SHAP TreeExplainer
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(sample_data)

        # Ensure correct SHAP values for class 1
        correct_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

        # SHAP Summary Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(correct_shap_values, sample_data, feature_names=feature_names, show=False)
        st.pyplot(fig)

        # SHAP Feature Importance Bar Chart (Fixed)
        st.subheader("🔹 SHAP Feature Importance (Bar Chart)")
        shap_importance = np.abs(correct_shap_values).mean(axis=0)
        importance_df = pd.DataFrame({'Feature': list(feature_names), 'SHAP Importance': shap_importance})

        # Sort and Plot
        importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.barh(importance_df["Feature"][:10], importance_df["SHAP Importance"][:10], color="skyblue")
        plt.xlabel("SHAP Importance")
        plt.ylabel("Feature")
        plt.title("Top 10 Important Features")
        plt.gca().invert_yaxis()
        st.pyplot(fig)
