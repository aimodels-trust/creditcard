import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Set page layout
st.set_page_config(page_title="Credit Default Prediction", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predictions", "📊 Feature Importance"])

# Step 1: Download the model from Google Drive (if not exists)
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(model_url, model_path, quiet=False)

# Step 2: Load trained model
model = joblib.load(model_path)

# Expected input features
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# ------------------------ HOME PAGE ------------------------
if app_mode == "🏠 Home":
    st.title("📌 Credit Default Prediction & Explainability")
    st.write("""
    Welcome to the **Credit Default Prediction** app!  
    This tool predicts if a customer will default on their credit card payment and provides an **explanation using SHAP values**.
    
    **How to use:**  
    1️⃣ Upload a CSV file with credit details.  
    2️⃣ View predictions and probability of default.  
    3️⃣ Explore feature importance and interpretability.  
    """)

# ------------------------ PREDICTIONS PAGE ------------------------
elif app_mode == "📈 Predictions":
    st.title("📈 Credit Default Predictions")
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV does not match expected format. Please check the number of columns.")
        else:
            with st.spinner("Processing data..."):
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

            # Display results
            st.success("✅ Predictions Complete!")
            st.write("### Prediction Results")
            st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

# ------------------------ FEATURE IMPORTANCE PAGE ------------------------
elif app_mode == "📊 Feature Importance":
    st.title("📊 Feature Importance & Explainability")

    uploaded_file = st.file_uploader("📂 Upload CSV file for SHAP Analysis", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV does not match expected format. Please check the number of columns.")
        else:
            with st.spinner("Processing data..."):
                # Extract preprocessor and classifier
                preprocessor = model.named_steps['preprocessor']
                classifier = model.named_steps['classifier']
                
                # Transform input data
                X_transformed = preprocessor.transform(df)

                # Get transformed feature names
                feature_names = preprocessor.get_feature_names_out()

                # Use a small sample for faster SHAP computation
                sample_data = X_transformed[:50]

                # SHAP explainability
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(sample_data)

            # Ensure SHAP values are correctly indexed
            correct_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

            # Ensure correct shapes
            if sample_data.shape[1] != correct_shap_values.shape[1]:
                st.error(f"Shape mismatch: Features = {sample_data.shape[1]}, SHAP Values = {correct_shap_values.shape[1]}")
            else:
                # Compute SHAP feature importance
                shap_importance = np.abs(correct_shap_values).mean(axis=0).flatten()
                feature_names = np.array(feature_names).flatten()

                if len(shap_importance) != len(feature_names):
                    st.error("Feature importance computation failed due to shape mismatch.")
                else:
                    # Create feature importance dataframe
                    importance_df = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
                    importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).head(10)

                    # Display feature importance table
                    st.write("### 🔥 Top 10 Most Important Features")
                    st.dataframe(importance_df)

                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="royalblue")
                    ax.set_xlabel("SHAP Importance")
                    ax.set_ylabel("Feature")
                    ax.set_title("📊 Feature Importance")
                    plt.gca().invert_yaxis()  # Invert to show highest importance at the top
                    st.pyplot(fig)

                    # SHAP Summary Plot
                    st.write("### 📊 SHAP Summary Plot")
                    shap.summary_plot(correct_shap_values, sample_data, feature_names=feature_names, show=False)
                    plt.savefig("shap_summary.png", bbox_inches='tight')
                    st.image("shap_summary.png")
