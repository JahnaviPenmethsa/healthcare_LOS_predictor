from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the API
app = FastAPI(title="Hospital LOS Prediction API")

# Load LOS Model
rf_model = joblib.load('rf_model.joblib')
rf_features = joblib.load('model_features.joblib')

# Load Risk Model
xgb_model = joblib.load('xgb_model.joblib')
xgb_features = joblib.load('classification_features.joblib')

# Load Respiratory Model
resp_model = joblib.load('resp_model.joblib')
resp_features = joblib.load('resp_features.joblib')
class PatientData(BaseModel):
    data: dict

@app.post("/predict")
def predict_los(patient: PatientData):
    input_df = pd.DataFrame([patient.data])
    for col in rf_features:
        if col not in input_df.columns:
            input_df[col] = 0
    prediction = rf_model.predict(input_df[rf_features])
    return {"predicted_length_of_stay_days": round(float(prediction[0]), 2)}

@app.post("/predict_risk")
def predict_risk(patient: PatientData):
    input_df = pd.DataFrame([patient.data])
    for col in xgb_features:
        if col not in input_df.columns:
            input_df[col] = 0
    # XGBoost returns probabilities for [Healthy, Sick]
    risk_prob = xgb_model.predict_proba(input_df[xgb_features])[0][1]
    return {"renal_risk_probability": round(float(risk_prob), 2)}

@app.post("/predict_respiratory")
def predict_respiratory(patient: PatientData):
    input_df = pd.DataFrame([patient.data])
    for col in resp_features:
        if col not in input_df.columns:
            input_df[col] = 0
    risk_prob = resp_model.predict_proba(input_df[resp_features])[0][1]
    return {"respiratory_risk_probability": round(float(risk_prob), 2)}