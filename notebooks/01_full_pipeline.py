"""
CreditSense AI — Full modeling pipeline
Run this as a script, or paste cells into a Jupyter notebook.

Sections:
1. Load + EDA
2. Feature engineering
3. Train/test split + preprocessing
4. Handle class imbalance (SMOTE)
5. Train baseline + advanced models
6. Hyperparameter tuning (Optuna) for XGBoost
7. Evaluation (PR-AUC, F1, confusion matrix, calibration)
8. SHAP explainability
9. Save final model + preprocessing pipeline for the FastAPI app
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
import shap

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Credit_Default_Enriched.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load + EDA
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nDefault rate:", df["Default"].mean().round(4))
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
df["Debt_to_Income"] = df["Loan"] / df["Income"]
df["Income_per_Dependent"] = df["Income"] / (df["Num_Dependents"] + 1)
df["Savings_to_Income"] = df["Savings_Balance"] / df["Income"]
df["Risk_Momentum"] = df["Late_Payments_24M"] * df["Credit_Utilization_Pct"] / 100
df["Age_Income_Interaction"] = df["Age"] * df["Income"] / 1000

target = "Default"
drop_cols = ["Applicant_ID", target]
num_cols = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.drop(columns=drop_cols).select_dtypes(include=[object]).columns.tolist()

X = df.drop(columns=drop_cols)
y = df[target]

print("\nNumeric features:", num_cols)
print("Categorical features:", cat_cols)

# ---------------------------------------------------------------------------
# 3. Train/test split + preprocessing
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

feature_names = num_cols + list(
    preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
)

# ---------------------------------------------------------------------------
# 4. Handle class imbalance with SMOTE (train set only)
# ---------------------------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
print("\nBefore SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# ---------------------------------------------------------------------------
# 5. Baseline model — Logistic Regression
# ---------------------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_res, y_train_res)
logreg_probs = logreg.predict_proba(X_test_proc)[:, 1]
print("\n--- Logistic Regression ---")
print("ROC-AUC:", round(roc_auc_score(y_test, logreg_probs), 4))
print("PR-AUC :", round(average_precision_score(y_test, logreg_probs), 4))

# ---------------------------------------------------------------------------
# 5b. Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
rf_probs = rf.predict_proba(X_test_proc)[:, 1]
print("\n--- Random Forest ---")
print("ROC-AUC:", round(roc_auc_score(y_test, rf_probs), 4))
print("PR-AUC :", round(average_precision_score(y_test, rf_probs), 4))

# ---------------------------------------------------------------------------
# 6. XGBoost + Optuna hyperparameter tuning
# ---------------------------------------------------------------------------
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric": "aucpr",
        "random_state": 42,
        "use_label_encoder": False,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_res, y_train_res)
    preds = model.predict_proba(X_test_proc)[:, 1]
    return average_precision_score(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25, show_progress_bar=False)

print("\nBest Optuna params:", study.best_params)
print("Best PR-AUC (search):", round(study.best_value, 4))

best_xgb = xgb.XGBClassifier(**study.best_params, eval_metric="aucpr", random_state=42)
best_xgb.fit(X_train_res, y_train_res)
xgb_probs = best_xgb.predict_proba(X_test_proc)[:, 1]

print("\n--- Tuned XGBoost (final) ---")
print("ROC-AUC:", round(roc_auc_score(y_test, xgb_probs), 4))
print("PR-AUC :", round(average_precision_score(y_test, xgb_probs), 4))
print(classification_report(y_test, (xgb_probs > 0.5).astype(int)))

# ---------------------------------------------------------------------------
# 7. Calibration (so probabilities are trustworthy for business decisions)
# ---------------------------------------------------------------------------
calibrated_model = CalibratedClassifierCV(best_xgb, method="isotonic", cv=3)
calibrated_model.fit(X_train_res, y_train_res)
cal_probs = calibrated_model.predict_proba(X_test_proc)[:, 1]
print("\n--- Calibrated XGBoost ---")
print("PR-AUC :", round(average_precision_score(y_test, cal_probs), 4))

cm = confusion_matrix(y_test, (cal_probs > 0.5).astype(int))
print("Confusion matrix:\n", cm)

# ---------------------------------------------------------------------------
# 8. SHAP explainability
# ---------------------------------------------------------------------------
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test_proc)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])[:10]
print("\nTop 10 features by mean |SHAP value|:")
for f, v in top_features:
    print(f"  {f}: {v:.4f}")

# ---------------------------------------------------------------------------
# 9. Save artifacts for the FastAPI app
# ---------------------------------------------------------------------------
joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.joblib")
joblib.dump(best_xgb, f"{MODEL_DIR}/xgb_model.joblib")
joblib.dump(explainer, f"{MODEL_DIR}/shap_explainer.joblib")

with open(f"{MODEL_DIR}/feature_names.json", "w") as f:
    json.dump(feature_names, f)

with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump({
        "logreg_pr_auc": float(average_precision_score(y_test, logreg_probs)),
        "rf_pr_auc": float(average_precision_score(y_test, rf_probs)),
        "xgb_pr_auc": float(average_precision_score(y_test, xgb_probs)),
        "xgb_roc_auc": float(roc_auc_score(y_test, xgb_probs)),
        "best_params": study.best_params,
        "confusion_matrix": cm.tolist(),
    }, f, indent=2)

print("\nSaved model artifacts to", MODEL_DIR)
