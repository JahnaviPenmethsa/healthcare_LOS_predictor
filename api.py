from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the API
app = FastAPI(title="Hospital LOS Prediction API")

# Load our saved model and features
model = joblib.load('rf_model.joblib')
expected_features = joblib.load('model_features.joblib')

# Define the expected incoming data format
class PatientData(BaseModel):
    data: dict

@app.post("/predict")
def predict_los(patient: PatientData):
    input_df = pd.DataFrame([patient.data])

    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_features]
    prediction = model.predict(input_df)

    return {"predicted_length_of_stay_days": round(float(prediction[0]), 2)}