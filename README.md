# Hospital Resource Optimization & Clinical Risk Predictor

A production-ready machine learning system built to predict patient hospital length of stay (LOS) and assess high-risk complications (End-Stage Renal Disease and Respiratory failure) at the time of admission. This multi-target tool aids hospital administration in bed management and assists clinicians in early patient triage.

## System Architecture

```mermaid
graph TD
    A[Hospital Staff] -->|Enters Patient Vitals| B(Streamlit UI)
    B -->|JSON POST: /predict| C{FastAPI Backend}
    B -->|JSON POST: /predict_risk| C
    B -->|JSON POST: /predict_respiratory| C
    C -->|Loads rf_model| D[(Random Forest: LOS)]
    C -->|Loads xgb_model| E[(XGBoost: Renal Risk)]
    C -->|Loads resp_model| F[(XGBoost: Respiratory Risk)]
    D --> C
    E --> C
    F --> C
    C -->|Returns Predictions| B
```

## Tech Stack

* **Language:** Python
* **Core ML:** Scikit-Learn, XGBoost, Pandas, NumPy, SHAP
* **Backend API:** FastAPI, Uvicorn, Render
* **Frontend UI:** Streamlit, Streamlit Cloud
* **Model Artifacts:** Joblib

## Key Features & Engineering Rigor

* **Modular Scalability (1-to-N):** Designed the FastAPI backend and Streamlit frontend to accept new disease models seamlessly, scaling from a single LOS regression model to a multi-model triage hub.
* **Data Leakage Prevention:** Explicitly dropped discharge dates and post-admission features to ensure real-world viability at the moment of admission.
* **Model Performance:** Compared Linear Regression baseline against a tuned Random Forest Regressor, improving R2 from 0.768 to 0.817 and reducing MAE to 0.77 days.
* **Handling Medical Imbalance:** Built XGBoost classifiers for Renal and Respiratory risks, utilizing scale_pos_weight to prioritize Recall (0.75) over plain accuracy, catching 89% more at-risk patients than the baseline.
* **Explainability:** Used SHAP values to identify prior visit counts (rcount) and facility type (facid_E) as primary drivers of extended stays.