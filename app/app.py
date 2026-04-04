# ---------------------------------------------
# Import required libraries
# ---------------------------------------------

import sys
import os
import streamlit as st
st.set_page_config(layout="wide")

# Fix path to access src folder
sys.path.append(os.path.abspath("."))

import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model from HF
model_path = hf_hub_download(
    repo_id="Karthickshiva07/engine-failure-model",
    filename="engine_failure_model.pkl"
)

model = joblib.load(model_path)

def predict_engine_condition(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return "Engine Failure Likely" if prediction == 1 else "Engine Healthy"


# ---------------------------------------------
# App Title
# ---------------------------------------------

st.title("Engine Predictive Maintenance System V1")

st.write("Enter engine sensor values to predict engine condition")


# ---------------------------------------------
# Input Fields
# ---------------------------------------------

# User inputs for each sensor
engine_rpm = st.number_input("Engine RPM", min_value=0.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure", min_value=0.0)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0)
coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", min_value=0.0)
coolant_temp = st.number_input("Coolant Temperature", min_value=0.0)


# ---------------------------------------------
# Prediction Button
# ---------------------------------------------

if st.button("Predict Engine Condition"):

    # Create input dictionary (IMPORTANT)
    input_data = {
        "Engine_RPM": engine_rpm,
        "Lub_Oil_Pressure": lub_oil_pressure,
        "Fuel_Pressure": fuel_pressure,
        "Coolant_Pressure": coolant_pressure,
        "Lub_Oil_Temperature": lub_oil_temp,
        "Coolant_Temperature": coolant_temp
    }

    # Call prediction function correctly
    result = predict_engine_condition(input_data)

    # Display result
    st.subheader("Prediction Result:")
    if "Failure" in result:
        st.error(result)
    else:
        st.success(result)
