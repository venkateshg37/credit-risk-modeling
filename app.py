import streamlit as st
import requests

st.title("Credit Risk Prediction")

# Inputs
credit_amount = st.number_input("Credit Amount")
duration = st.number_input("Loan Duration (months)")
age = st.number_input("Age")

# Feature engineering SAME as training
credit_utilization = credit_amount / 10000  # keep consistent with training

if st.button("Predict Risk"):
    data = {
        "CreditAmount": credit_amount,
        "Duration": duration,
        "Age": age,
        "credit_utilization": credit_utilization
    }

    res = requests.post("http://127.0.0.1:8000/predict", json=data)

    if res.status_code == 200:
        result = res.json()
        st.success(f"Risk: {result['risk']}")
        st.write(f"Probability: {result['probability']:.2f}")
    else:
        st.error("Error occurred")