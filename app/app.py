# ---------------------------------------------
# Import required libraries
# ---------------------------------------------

import sys
import os

# Fix path to access src folder
sys.path.append(os.path.abspath("."))

import streamlit as st                     # UI framework
from src.predict import predict_engine_condition  # Prediction function


# ---------------------------------------------
# App Title
# ---------------------------------------------

st.title("Engine Predictive Maintenance System")

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

    # Call prediction function
    result = predict_engine_condition(
        engine_rpm,
        lub_oil_pressure,
        fuel_pressure,
        coolant_pressure,
        lub_oil_temp,
        coolant_temp
    )

    # Display result
    st.subheader("Prediction Result:")
    if "Failure" in result:
        st.error(result)
    else:
        st.success(result)
