import pickle
import streamlit as st

st.title("🏦 Loan Approval Predictor")

income = st.number_input("Applicant Income ($)", value=None)
credit_score = st.number_input("Credit Score", value=None)
dti = st.number_input("DTI Ratio (%)", value=None)
loans = st.number_input("Existing Loans", value=None)
loan_amount = st.number_input("Loan Amount ($)", value=None)

if st.button("Predict"):
    with open("model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    data = scaler.transform([[income, credit_score, dti, loans, loan_amount]])
    result = model.predict(data)[0]
    if result == 1:
        st.success("✅ Approved")
    else:
        st.error("❌ Rejected")
