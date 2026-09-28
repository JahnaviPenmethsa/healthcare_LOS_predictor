import streamlit as st
import requests

st.title("🏥 Hospital Length of Stay Predictor")
st.write("Enter patient details at admission to predict how many days they will need a bed.")

# 1. Create input fields for the user
col1, col2 = st.columns(2)
with col1:
    prior_visits = st.selectbox("Prior Visits (rcount)", ["0", "1", "2", "3", "4", "5+"])
with col2:
    facility = st.selectbox("Facility ID", ["A", "B", "C", "D", "E"])
    
hematocrit = st.slider("Hematocrit Level", 10.0, 20.0, 12.0)

# 2. Predict button
if st.button("Predict LOS"):
    # 3. Format the data to match what the API expects
    data = {"hematocrit": hematocrit}
    
    if prior_visits != "0":
        data[f"rcount_{prior_visits}"] = 1
    data[f"facid_{facility}"] = 1
    
    # 4. Send the data to our running FastAPI backend
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json={"data": data})
        if response.status_code == 200:
            result = response.json()
            st.success(f"**Predicted Length of Stay:** {result['predicted_length_of_stay_days']} days")
        else:
            st.error("API Error")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Is Uvicorn running in your other terminal?")