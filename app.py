import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# -------------------------
# Step 1: Download the model from Google Drive
# -------------------------
# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1lFMD3ZSGuj72cprZ6_EOoos68l2KZxvP"
model_path = "credit_default_model.pkl"

@st.cache_resource  # Caches model loading to avoid reloading on every interaction
def load_model():
    if not os.path.exists(model_path):
        st.info("Downloading model from Google Drive...")
        gdown.download(model_url, model_path, quiet=False)

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

# -------------------------
# Step 2: Build the Streamlit UI
# -------------------------

st.title("💳 Credit Card Default Prediction App")
st.markdown("This app predicts whether a person will default on their credit card payment based on financial and demographic inputs.")

st.header("📌 Enter Customer Details")

# User Inputs
limit_bal = st.number_input("Credit Limit ($)", min_value=1000, step=500, value=50000)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
sex = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
marriage = st.selectbox("Marital Status", ["Married", "Single", "Others"])
pay_0 = st.selectbox("Repayment Status (Most Recent Month)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Financial Inputs
st.subheader("📊 Financial Details")
bill_amt1 = st.number_input("Bill Amount (Last Month) ($)", min_value=0, value=20000)
pay_amt1 = st.number_input("Payment Amount (Last Month) ($)", min_value=0, value=5000)

# -------------------------
# Step 3: Make Prediction
# -------------------------

if st.button("🔍 Predict Default Risk"):
    if model is None:
        st.error("❌ Model could not be loaded. Please check the model file.")
    else:
        try:
            # Map categorical values to numerical
            sex_mapping = {'Male': 1, 'Female': 2}
            education_mapping = {'Graduate School': 1, 'University': 2, 'High School': 3, 'Others': 4}
            marriage_mapping = {'Married': 1, 'Single': 2, 'Others': 3}

            # Prepare input data
            input_data = pd.DataFrame({
                'LIMIT_BAL': [limit_bal],
                'AGE': [age],
                'SEX': [sex_mapping[sex]],
                'EDUCATION': [education_mapping[education]],
                'MARRIAGE': [marriage_mapping[marriage]],
                'PAY_0': [pay_0],
                'PAY_2': [pay_0], 'PAY_3': [pay_0], 'PAY_4': [pay_0], 'PAY_5': [pay_0], 'PAY_6': [pay_0],
                'BILL_AMT1': [bill_amt1], 'BILL_AMT2': [bill_amt1 * 0.9], 'BILL_AMT3': [bill_amt1 * 0.8],
                'BILL_AMT4': [bill_amt1 * 0.7], 'BILL_AMT5': [bill_amt1 * 0.6], 'BILL_AMT6': [bill_amt1 * 0.5],
                'PAY_AMT1': [pay_amt1], 'PAY_AMT2': [pay_amt1 * 0.9], 'PAY_AMT3': [pay_amt1 * 0.8],
                'PAY_AMT4': [pay_amt1 * 0.7], 'PAY_AMT5': [pay_amt1 * 0.6], 'PAY_AMT6': [pay_amt1 * 0.5]
            })

            # Ensure the input matches model expectations
            expected_columns = [
                'LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE',
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
            ]
            input_data = input_data[expected_columns]

            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            # Display results
            st.subheader("📢 Prediction Result:")
            if prediction == 1:
                st.error(f"🚨 **High Default Risk** ({probability:.2%})")
            else:
                st.success(f"✅ **Low Default Risk** ({probability:.2%})")

        except Exception as e:
            st.error(f"⚠️ An error occurred while making the prediction: {e}")
