import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Default Prediction", layout="wide")

# Download model if not present
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    st.info("Downloading model... Please wait ⏳")
    gdown.download(model_url, model_path, quiet=False)

# Load trained model
model = joblib.load(model_path)

st.title("🔍 Credit Default Prediction with Explainability")

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
        st.error("❌ Uploaded CSV does not match expected format.")
    else:
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        X_transformed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out()

        predictions = classifier.predict(X_transformed)
        probabilities = classifier.predict_proba(X_transformed)[:, 1]

        df['Default_Risk'] = predictions
        df['Probability'] = probabilities

        st.subheader("📊 Prediction Results")
        st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

        # Optimize SHAP Computation
        st.subheader("📈 Feature Importance & Explainability")

        sample_data = X_transformed[:20]  # Reduce sample size

        @st.cache_resource
        def compute_shap_values(sample_data):
            explainer = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
            return explainer.shap_values(sample_data, approximate=True)

        shap_values = compute_shap_values(sample_data)

        correct_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Feature Importance Bar Chart
        num_top_features = 10  # Limit to top 10 features
        shap_importance = np.abs(correct_shap_values).mean(axis=0)
        importance_df = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
        importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).head(num_top_features)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="skyblue")
        plt.xlabel("SHAP Importance")
        plt.ylabel("Feature")
        plt.title("Top 10 Important Features")
        plt.gca().invert_yaxis()
        st.pyplot(fig)
