"""
train_model.py
---------------
Trains a Logistic Regression and Random Forest for BOTH kidney and liver
disease risk, keeps whichever scores higher on ROC-AUC for each, and saves
them to kidney_model.joblib / liver_model.joblib.

Run:
    python generate_data.py
    python train_model.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KIDNEY_FEATURES = [
    "age", "sex", "systolic_bp", "diastolic_bp", "creatinine", "egfr", "bun",
    "sodium", "potassium", "hemoglobin", "urine_acr", "hba1c", "diabetes",
    "hypertension", "bmi", "smoker", "family_history",
]
LIVER_FEATURES = [
    "age", "sex", "bmi", "alt", "ast", "alp", "ggt", "bilirubin", "albumin",
    "total_protein", "platelets", "inr", "diabetes", "alcohol_units_per_week",
    "family_history",
]


def train_one(csv_path, features, model_path, label):
    df = pd.read_csv(csv_path)
    X, y = df[features], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
            )),
        ]),
    }

    results = {}
    print(f"\n########## {label} ##########")
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = pipe.predict(X_test)
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}   ROC-AUC: {auc:.3f}")
        print(classification_report(y_test, preds, target_names=["low_risk", "high_risk"]))
        results[name] = {"pipe": pipe, "auc": auc}

    # Prefer logistic regression unless a tree model beats it by a clear
    # margin: sliders in this app let users push values well outside the
    # training distribution, and random forests don't extrapolate past
    # their training range (predictions flatten toward the base rate for
    # out-of-range inputs), which produces misleadingly muted risk scores
    # for very severe profiles. Logistic regression stays monotonic and
    # responsive across the whole slider range, so it's the safer default
    # for an interactive "what happens if I move this slider" tool.
    lr_auc = results["logistic_regression"]["auc"]
    rf_auc = results["random_forest"]["auc"]
    if rf_auc > lr_auc + 0.05:
        best_name = "random_forest"
    else:
        best_name = "logistic_regression"
    best_pipe = results[best_name]["pipe"]
    best_auc = results[best_name]["auc"]

    print(f"\nBest model for {label}: {best_name} (ROC-AUC={best_auc:.3f})")
    joblib.dump(
        {"pipeline": best_pipe, "features": features, "model_name": best_name},
        model_path,
    )
    print(f"Saved to {model_path}")


if __name__ == "__main__":
    train_one("kidney_data.csv", KIDNEY_FEATURES, "kidney_model.joblib", "Kidney disease")
    train_one("liver_data.csv", LIVER_FEATURES, "liver_model.joblib", "Liver disease")
