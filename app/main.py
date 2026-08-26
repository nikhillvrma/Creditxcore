"""
Creditxcore — FastAPI backend
Serves credit-default risk predictions with SHAP-based per-applicant explanations.

Run locally:
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")

app = FastAPI(
    title="Creditxcore",
    description="Explainable credit default risk scoring API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")
model = joblib.load(f"{MODEL_DIR}/xgb_model.joblib")
explainer = joblib.load(f"{MODEL_DIR}/shap_explainer.joblib")
with open(f"{MODEL_DIR}/feature_names.json") as f:
    feature_names = json.load(f)


class Applicant(BaseModel):
    Income: float = Field(..., example=45000.0)
    Age: float = Field(..., example=35.0)
    Loan: float = Field(..., example=5000.0)
    Loan_to_Income: float = Field(..., example=0.11, alias="Loan to Income")
    Employment_Type: str = Field(..., example="Salaried")
    Credit_History_Years: float = Field(..., example=8.0)
    Num_Existing_Loans: int = Field(..., example=1)
    Late_Payments_24M: int = Field(..., example=0)
    Credit_Utilization_Pct: float = Field(..., example=30.0)
    EMI_Burden_Ratio_Pct: float = Field(..., example=15.0)
    Num_Dependents: int = Field(..., example=1)
    City_Tier: str = Field(..., example="Tier-1")
    Savings_Balance: float = Field(..., example=8000.0)

    class Config:
        populate_by_name = True


def build_feature_row(a: Applicant) -> pd.DataFrame:
    row = {
        "Income": a.Income,
        "Age": a.Age,
        "Loan": a.Loan,
        "Loan to Income": a.Loan_to_Income,
        "Employment_Type": a.Employment_Type,
        "Credit_History_Years": a.Credit_History_Years,
        "Num_Existing_Loans": a.Num_Existing_Loans,
        "Late_Payments_24M": a.Late_Payments_24M,
        "Credit_Utilization_Pct": a.Credit_Utilization_Pct,
        "EMI_Burden_Ratio_Pct": a.EMI_Burden_Ratio_Pct,
        "Num_Dependents": a.Num_Dependents,
        "City_Tier": a.City_Tier,
        "Savings_Balance": a.Savings_Balance,
    }
    df = pd.DataFrame([row])
    df["Debt_to_Income"] = df["Loan"] / df["Income"]
    df["Income_per_Dependent"] = df["Income"] / (df["Num_Dependents"] + 1)
    df["Savings_to_Income"] = df["Savings_Balance"] / df["Income"]
    df["Risk_Momentum"] = df["Late_Payments_24M"] * df["Credit_Utilization_Pct"] / 100
    df["Age_Income_Interaction"] = df["Age"] * df["Income"] / 1000
    return df


@app.get("/")
def root():
    return {"status": "ok", "service": "CreditSense AI"}


@app.post("/predict")
def predict(applicant: Applicant):
    df = build_feature_row(applicant)
    X = preprocessor.transform(df)

    proba = float(model.predict_proba(X)[0, 1])
    decision = "HIGH RISK - Recommend Manual Review" if proba > 0.5 else "LOW RISK - Approve"

    shap_vals = explainer.shap_values(X)[0]
    contributions = sorted(
        zip(feature_names, shap_vals.tolist()), key=lambda x: -abs(x[1])
    )[:5]

    explanation = [
        {
            "feature": f,
            "impact": round(v, 4),
            "direction": "increases risk" if v > 0 else "decreases risk",
        }
        for f, v in contributions
    ]

    return {
        "default_probability": round(proba, 4),
        "decision": decision,
        "top_reasons": explanation,
    }
