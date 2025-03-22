import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Step 1: Page Configuration
st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide"
)

# Step 2: Add Stylish Header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Credit Card Default Prediction with Explainability</h1>",
    unsafe_allow_html=True
)

# Step 3: Download Model from Google Drive
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    with st.spinner("Downloading model... Please wait."):
        gdown.download(model_url, model_path, quiet=False)

# Step 4: Load the trained model
model = joblib.load(model_path)

# Step 5: Sidebar for File Upload
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Define expected input features
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Step 6: Main Logic
if uploaded_file is not None:
    st.sidebar.success("✅ File Uploaded Successfully!")

    df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

    if df.shape[1] != len(expected_columns):
        st.error("❌ Uploaded CSV does not match expected format. Please check the number of columns.")
    else:
        # Extract preprocessor and classifier
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Transform input data
        X_transformed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out()

        # Make predictions
        predictions = classifier.predict(X_transformed)
        probabilities = classifier.predict_proba(X_transformed)[:, 1]

        df['Default_Risk'] = predictions
        df['Probability'] = probabilities

        # Step 7: Display Results in a Table
        st.subheader("📊 Prediction Results")
        st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

        # Step 8: SHAP Explainability
        st.subheader("📌 Feature Importance (Explainability)")
        
        # Use a small sample for SHAP
        sample_data = X_transformed[:50]
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(sample_data)

        # Debugging Info
        st.write(f"🔹 Sample Data Shape: {sample_data.shape}")
        st.write(f"🔹 SHAP Values Shape: {shap_values[1].shape}")
        st.write(f"🔹 Feature Names Count: {len(feature_names)}")

        # Ensure correct SHAP values for class 1
        correct_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

        if sample_data.shape[1] != correct_shap_values.shape[1]:
            st.error(f"⚠️ Shape mismatch: Features = {sample_data.shape[1]}, SHAP Values = {correct_shap_values.shape[1]}")
        else:
            # SHAP Summary Plot
            shap.summary_plot(correct_shap_values, sample_data, feature_names=feature_names, show=False)
            plt.savefig("shap_summary.png", bbox_inches='tight')
            st.image("shap_summary.png", caption="Feature Importance Summary")

            # Step 9: Generate Individual Feature Importance Plots
            st.subheader("🔍 Top Feature Analysis")

            top_features = np.argsort(np.abs(correct_shap_values).mean(axis=0))[-5:][::-1]  # Get top 5 features

            for feature_idx in top_features:
                feature_name = feature_names[feature_idx]
                st.write(f"### {feature_name} Impact")
                shap.dependence_plot(feature_name, correct_shap_values, sample_data, show=False)
                plt.savefig(f"shap_{feature_name}.png", bbox_inches='tight')
                st.image(f"shap_{feature_name}.png")

# Step 10: Add Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #4CAF50;
        text-align: center;
        padding: 5px;
        font-size: 14px;
        color: white;
    }
    </style>
    <div class='footer'>Developed by AI/ML Senior Design Team 🚀</div>
    """,
    unsafe_allow_html=True
)
