"""
Enriches the original Credit_Default.csv (Income, Age, Loan, Loan to Income, Default)
with additional realistic features so the dataset supports a genuinely advanced
credit-risk modeling project instead of a bare 4-feature toy dataset.

IMPORTANT: none of the new columns are derived from `Default`. They are derived
only from Income / Age / Loan plus independent random noise, exactly like a real
data-enrichment exercise would work (you'd never have access to the future
default outcome when building applicant features). This keeps the dataset free
of target leakage.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Resolve paths relative to this script's location, so it works no matter
# which machine / folder the project is run from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "Credit_Default.csv")

df = pd.read_csv(INPUT_CSV)
n = len(df)

income = df["Income"].values
age = df["Age"].values
loan = df["Loan"].values

# 1. Employment type — mildly linked to income percentile, otherwise random
income_pct = pd.Series(income).rank(pct=True).values
employment_type = np.where(
    income_pct > 0.66,
    np.random.choice(["Salaried", "Self-Employed"], n, p=[0.7, 0.3]),
    np.where(
        income_pct > 0.33,
        np.random.choice(["Salaried", "Self-Employed", "Contract"], n, p=[0.5, 0.25, 0.25]),
        np.random.choice(["Contract", "Self-Employed", "Salaried"], n, p=[0.45, 0.30, 0.25]),
    ),
)

# 2. Credit history length (years) — correlates with Age + noise
credit_history_years = np.round(
    np.clip((age - 18) * np.random.uniform(0.15, 0.75, n), 0, None), 1
)

# 3. Number of existing loans — mild link to loan size + independent noise
loan_pct = pd.Series(loan).rank(pct=True).values
num_existing_loans = np.clip(
    np.random.poisson(lam=0.8 + loan_pct * 0.8, size=n), 0, 8
)

# 4. Number of late payments in last 24 months — mostly independent noise
late_payments_24m = np.clip(np.random.poisson(lam=0.7, size=n), 0, 15)

# 5. Credit utilization ratio (%) — independent-ish, income has mild damping effect
credit_utilization = np.clip(
    np.random.beta(2, 3, n) * 100 - (income_pct - 0.5) * 10, 0, 100
).round(2)

# 6. Monthly EMI burden ratio (% of income) — derived from loan/income + noise only
emi_burden_ratio = np.clip(
    (loan / income) * np.random.uniform(0.08, 0.16, n) * 100, 1, 90
).round(2)

# 7. Number of dependents — independent
num_dependents = np.random.choice([0, 1, 2, 3, 4], n, p=[0.30, 0.28, 0.24, 0.12, 0.06])

# 8. Region / city tier — independent (good for a fairness/bias audit angle)
city_tier = np.random.choice(["Tier-1", "Tier-2", "Tier-3"], n, p=[0.42, 0.35, 0.23])

# 9. Savings account balance — scales with income + noise only
savings_balance = np.clip(income * np.random.uniform(0.02, 0.25, n), 0, None).round(2)

enriched = df.copy()
enriched.insert(0, "Applicant_ID", [f"APP{100000+i}" for i in range(n)])
enriched["Employment_Type"] = employment_type
enriched["Credit_History_Years"] = credit_history_years
enriched["Num_Existing_Loans"] = num_existing_loans
enriched["Late_Payments_24M"] = late_payments_24m
enriched["Credit_Utilization_Pct"] = credit_utilization
enriched["EMI_Burden_Ratio_Pct"] = emi_burden_ratio
enriched["Num_Dependents"] = num_dependents
enriched["City_Tier"] = city_tier
enriched["Savings_Balance"] = savings_balance

cols = [c for c in enriched.columns if c != "Default"] + ["Default"]
enriched = enriched[cols]

out_path = os.path.join(SCRIPT_DIR, "Credit_Default_Enriched.csv")
enriched.to_csv(out_path, index=False)
print(f"Saved enriched dataset: {enriched.shape} -> {out_path}")
print(enriched.head(3).to_string())
