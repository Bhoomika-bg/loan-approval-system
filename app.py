import pickle
import streamlit as st

st.title("🏦 Loan Approval Predictor")

# Inputs with proper ranges
income = st.number_input("Applicant Income ($)", min_value=1000, max_value=500000, value=None)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=None)
dti = st.number_input("DTI Ratio (%)", min_value=1, max_value=100, value=None)
loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=None)
loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=None)

if st.button("Predict"):
    # Check if all inputs are filled
    if None in [income, credit_score, dti, loans, loan_amount]:
        st.warning("⚠️ Please fill all fields")
    else:
        # Load model
        with open("model.pkl", "rb") as f:
            model, scaler = pickle.load(f)

        # Transform input
        data = scaler.transform([[income, credit_score, dti, loans, loan_amount]])

        # Prediction
        result = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        # Output
        if result == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # Show probability
        st.write(f"📊 Approval Probability: {prob * 100:.2f}%")

        # Simple explanation
        if credit_score < 600:
            st.write("⚠️ Low credit score affects approval")
        if dti > 50:
            st.write("⚠️ High DTI ratio indicates higher risk")
