import streamlit as st
import requests

st.title("🏥 Hospital AI Assistant")

# Create two tabs
tab1, tab2 = st.tabs(["🛏️ Length of Stay", "⚠️ Renal Disease Risk"])

with tab1:
    st.write("Predict how many days a patient will need a bed.")
    col1, col2 = st.columns(2)
    with col1:
        prior_visits = st.selectbox("Prior Visits", ["0", "1", "2", "3", "4", "5+"])
    with col2:
        facility = st.selectbox("Facility ID", ["A", "B", "C", "D", "E"])
    hematocrit = st.slider("Hematocrit Level", 10.0, 20.0, 12.0)

    if st.button("Predict LOS"):
        data = {"hematocrit": hematocrit}
        if prior_visits != "0": data[f"rcount_{prior_visits}"] = 1
        data[f"facid_{facility}"] = 1
        
        response = requests.post("https://healthcare-los-predictor.onrender.com/predict", json={"data": data})
        st.success(f"**Predicted Length of Stay:** {response.json()['predicted_length_of_stay_days']} days")

with tab2:
    st.write("Predict the risk of End-Stage Renal Disease at admission.")
    blood_urea = st.slider("Blood Urea Nitrogen", 0.0, 50.0, 15.0)
    creatinine = st.slider("Creatinine Level", 0.0, 10.0, 1.0)
    
    if st.button("Predict Risk"):
        data = {"bloodureanitro": blood_urea, "creatinine": creatinine}
        response = requests.post("https://healthcare-los-predictor.onrender.com/predict_risk", json={"data": data})
        risk = response.json()['renal_risk_probability'] * 100
        
        if risk > 50:
            st.error(f"**High Risk:** {risk:.1f}% probability of Renal Disease")
        else:
            st.success(f"**Low Risk:** {risk:.1f}% probability")