"""
CreditSense AI — Streamlit demo UI
Run: streamlit run streamlit_app.py
(Make sure the FastAPI backend is running on http://127.0.0.1:8000 first)
"""

import streamlit as st
import requests

st.set_page_config(page_title="CreditSense AI", page_icon="💳")
st.title("💳 CreditSense AI — Credit Risk Assessment")
st.caption("Explainable credit-default risk scoring, powered by XGBoost + SHAP")

API_URL = "http://127.0.0.1:8000/predict"

with st.form("applicant_form"):
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income", 20000.0, 70000.0, 45000.0)
        age = st.number_input("Age", 18.0, 70.0, 35.0)
        loan = st.number_input("Loan Amount", 100.0, 15000.0, 5000.0)
        employment = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Contract"])
        credit_history = st.number_input("Credit History (years)", 0.0, 40.0, 8.0)
        num_loans = st.number_input("Existing Loans", 0, 10, 1)
        late_payments = st.number_input("Late Payments (24m)", 0, 20, 0)
    with col2:
        utilization = st.slider("Credit Utilization (%)", 0.0, 100.0, 30.0)
        emi_burden = st.slider("EMI Burden (% of income)", 1.0, 90.0, 15.0)
        dependents = st.number_input("Number of Dependents", 0, 6, 1)
        city_tier = st.selectbox("City Tier", ["Tier-1", "Tier-2", "Tier-3"])
        savings = st.number_input("Savings Balance", 0.0, 50000.0, 8000.0)

    submitted = st.form_submit_button("Assess Risk")

if submitted:
    payload = {
        "Income": income, "Age": age, "Loan": loan,
        "Loan to Income": round(loan / income, 4),
        "Employment_Type": employment, "Credit_History_Years": credit_history,
        "Num_Existing_Loans": int(num_loans), "Late_Payments_24M": int(late_payments),
        "Credit_Utilization_Pct": utilization, "EMI_Burden_Ratio_Pct": emi_burden,
        "Num_Dependents": int(dependents), "City_Tier": city_tier,
        "Savings_Balance": savings,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()

        st.subheader(f"Default Probability: {result['default_probability']*100:.2f}%")
        if "HIGH" in result["decision"]:
            st.error(result["decision"])
        else:
            st.success(result["decision"])

        st.write("**Top contributing factors:**")
        for reason in result["top_reasons"]:
            st.write(f"- `{reason['feature']}` {reason['direction']} (impact: {reason['impact']})")
    except Exception as e:
        st.error(f"Could not reach API: {e}. Make sure `uvicorn main:app` is running.")
