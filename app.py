import pickle
import streamlit as st

st.title("🏦 Loan Approval Predictor")

income       = st.number_input("Applicant Income ($)", 0)
credit_score = st.number_input("Credit Score", 0)
dti          = st.number_input("DTI Ratio (%)", 0)
loans        = st.number_input("Existing Loans", 0)
loan_amount = st.number_input("Loan Amount ($)", 0)

if st.button("Predict"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    result = model.predict([[income, credit_score, dti, loans, loan_amount]])[0]
    st.success("✅ Approved") if result == 1 else st.error("❌ Rejected")
