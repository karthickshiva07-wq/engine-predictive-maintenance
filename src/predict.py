# ---------------------------------------------
# Import required libraries
# ---------------------------------------------

import joblib                         # Load trained model
import pandas as pd                  # Data formatting

# ---------------------------------------------
# Load trained model once (efficient)
# ---------------------------------------------

# Load model from model folder
from huggingface_hub import hf_hub_download
import joblib

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="Karthickshiva07/engine-failure-model",
    filename="engine_failure_model.pkl"
)

# Load model
model = joblib.load(model_path)

# ---------------------------------------------
# Prediction Function
# ---------------------------------------------

def predict_engine_condition(
    Engine_RPM,
    Lub_Oil_Pressure,
    Fuel_Pressure,
    Coolant_Pressure,
    Lub_Oil_Temperature,
    Coolant_Temperature
):
    """
    Takes sensor inputs and returns prediction
    """

    # Convert input into DataFrame (same structure as training)
    input_data = pd.DataFrame([{
        "Engine_RPM": Engine_RPM,
        "Lub_Oil_Pressure": Lub_Oil_Pressure,
        "Fuel_Pressure": Fuel_Pressure,
        "Coolant_Pressure": Coolant_Pressure,
        "Lub_Oil_Temperature": Lub_Oil_Temperature,
        "Coolant_Temperature": Coolant_Temperature
    }])

    # Perform prediction
    prediction = model.predict(input_data)[0]

    # Convert numeric output to meaningful label
    if prediction == 1:
        return "Engine Failure Likely"
    else:
        return "Engine Healthy"
    
from src.predict import predict_engine_condition

result = predict_engine_condition(
    800, 3.2, 7.5, 2.5, 78, 85
)
print(result)