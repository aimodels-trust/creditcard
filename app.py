import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model("credit_default_ann.h5")
scaler = StandardScaler()

# Define feature columns
features = ["LIMIT_BAL", "AVG_PAY_DELAY", "AVG_BILL_DIFF", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

def preprocess_input(data):
    """Preprocess input data by scaling."""
    data = scaler.transform(data)
    return data

def predict_single(input_data):
    """Make a single prediction."""
    processed_data = preprocess_input(np.array([input_data]))
    prob = model.predict(processed_data)[0][0]
    return prob, "Default" if prob > 0.5 else "No Default"

def predict_batch(uploaded_file):
    """Predict on batch CSV file."""
    df = pd.read_csv(uploaded_file)
    df_processed = preprocess_input(df[features])
    predictions = model.predict(df_processed)
    df['Default Probability'] = predictions
    df['Prediction'] = df['Default Probability'].apply(lambda x: "Default" if x > 0.5 else "No Default")
    return df

# Streamlit UI
st.title("Credit Card Default Prediction")

# Single Input Prediction
st.header("Single Prediction")
input_values = []
for feature in features:
    input_values.append(st.number_input(f"Enter {feature}", value=0.0))

if st.button("Predict Single Case"):
    prob, status = predict_single(input_values)
    st.write(f"Default Probability: {prob:.2f}")
    st.write(f"Prediction: {status}")

# Batch Prediction
st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    predictions_df = predict_batch(uploaded_file)
    st.write(predictions_df)
    st.download_button("Download Predictions", predictions_df.to_csv(index=False), "predictions.csv", "text/csv")
