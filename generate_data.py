"""
generate_data.py
-----------------
Creates two SYNTHETIC datasets:
  - kidney_data.csv  (chronic kidney disease risk)
  - liver_data.csv   (liver disease risk)

Both are generated from probabilistic models built around well-known
clinical risk factors (eGFR, creatinine, ACR, ALT/AST, bilirubin, etc.)
so the accompanying app has something realistic to train on.

This is NOT real patient data. Swap these files out for a real, ethically
sourced, properly licensed clinical dataset for anything beyond education.
"""

import numpy as np
import pandas as pd

SEED = 42
N = 4000


def generate_kidney(n=N, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(52, 15, n).clip(18, 90)
    sex = rng.integers(0, 2, n)  # 1 = male, 0 = female

    diabetes = (rng.random(n) < 0.22).astype(int)
    hypertension = (rng.random(n) < 0.30).astype(int)

    bmi = (26 + (age - 45) * 0.02 + rng.normal(0, 4.5, n)).clip(15, 50)

    systolic_bp = (118 + hypertension * 22 + (age - 45) * 0.35 + rng.normal(0, 10, n)).clip(90, 210)
    diastolic_bp = (76 + hypertension * 10 + rng.normal(0, 7, n)).clip(55, 130)

    hba1c = (5.2 + diabetes * 2.0 + rng.normal(0, 0.6, n)).clip(4.0, 13)

    # eGFR falls with age, hypertension, diabetes
    egfr = (
        100
        - 0.55 * (age - 30).clip(0, None)
        - 10 * hypertension
        - 9 * diabetes
        + rng.normal(0, 12, n)
    ).clip(5, 130)

    creatinine = (0.7 + (100 - egfr) * 0.018 + rng.normal(0, 0.15, n)).clip(0.4, 8)
    bun = (12 + (100 - egfr) * 0.35 + rng.normal(0, 4, n)).clip(5, 100)

    sodium = rng.normal(140, 3.5, n).clip(125, 152)
    potassium = (4.2 + (100 - egfr) * 0.01 + rng.normal(0, 0.35, n)).clip(2.8, 7.5)
    hemoglobin = (14.5 - (100 - egfr) * 0.03 - sex.astype(float) * -1.2 + rng.normal(0, 1.2, n)).clip(6, 18)

    urine_acr = np.exp(rng.normal(2.0 + diabetes * 1.1 + hypertension * 0.6, 1.1, n)).clip(1, 3000)

    family_history = (rng.random(n) < 0.18).astype(int)
    smoker = (rng.random(n) < 0.24).astype(int)

    z = (
        -5.5
        + 0.035 * age
        + 0.05 * (100 - egfr)
        + 0.9 * (egfr < 60)
        + 0.7 * (creatinine > 1.3)
        + 0.8 * (urine_acr >= 30)
        + 0.5 * (systolic_bp > 140)
        + 0.4 * (diastolic_bp > 90)
        + 0.6 * (hba1c >= 6.5)
        + 0.35 * (age > 60)
        + 0.3 * smoker
        + 0.35 * family_history
        + 0.02 * (bmi - 25)
        + rng.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-z))
    target = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age.round(0).astype(int),
        "sex": sex,
        "systolic_bp": systolic_bp.round(0).astype(int),
        "diastolic_bp": diastolic_bp.round(0).astype(int),
        "creatinine": creatinine.round(2),
        "egfr": egfr.round(1),
        "bun": bun.round(1),
        "sodium": sodium.round(1),
        "potassium": potassium.round(2),
        "hemoglobin": hemoglobin.round(1),
        "urine_acr": urine_acr.round(1),
        "hba1c": hba1c.round(1),
        "diabetes": diabetes,
        "hypertension": hypertension,
        "bmi": bmi.round(1),
        "smoker": smoker,
        "family_history": family_history,
        "target": target,
    })


def generate_liver(n=N, seed=SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(48, 14, n).clip(18, 90)
    sex = rng.integers(0, 2, n)  # 1 = male, 0 = female

    bmi = (26.5 + (age - 45) * 0.02 + rng.normal(0, 4.8, n)).clip(15, 50)
    diabetes = (rng.random(n) < 0.20).astype(int)
    alcohol_units_per_week = rng.gamma(1.4, 5.5, n).clip(0, 90)
    heavy_alcohol = (alcohol_units_per_week > 21).astype(int)

    liver_stress = (
        0.02 * (bmi - 25).clip(0, None)
        + 0.35 * diabetes
        + 0.03 * alcohol_units_per_week
        + rng.normal(0, 0.5, n)
    ).clip(0, None)

    alt = (22 + liver_stress * 14 + rng.normal(0, 8, n)).clip(5, 400)
    ast = (24 + liver_stress * 12 + rng.normal(0, 8, n)).clip(5, 400)
    alp = (75 + liver_stress * 8 + rng.normal(0, 20, n)).clip(30, 400)
    ggt = (28 + liver_stress * 20 + rng.normal(0, 15, n)).clip(5, 500)
    bilirubin = (0.7 + liver_stress * 0.35 + rng.normal(0, 0.3, n)).clip(0.1, 15)
    albumin = (4.3 - liver_stress * 0.18 + rng.normal(0, 0.3, n)).clip(1.5, 5.5)
    total_protein = (7.1 - liver_stress * 0.05 + rng.normal(0, 0.4, n)).clip(4, 9.5)
    platelets = (260 - liver_stress * 18 + rng.normal(0, 45, n)).clip(30, 500)
    inr = (1.0 + liver_stress * 0.06 + rng.normal(0, 0.08, n)).clip(0.8, 4)

    family_history = (rng.random(n) < 0.15).astype(int)

    z = (
        -4.0
        + 0.02 * age
        + 0.3 * sex
        + 0.03 * (bmi - 25).clip(0, None)
        + 0.6 * (alt > 40)
        + 0.6 * (ast > 40)
        + 0.7 * (bilirubin > 1.2)
        + 0.6 * (albumin < 3.5)
        + 0.5 * (inr > 1.2)
        + 0.5 * (bmi > 30)
        + 0.45 * diabetes
        + 0.55 * heavy_alcohol
        + 0.3 * family_history
        + rng.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-z))
    target = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age.round(0).astype(int),
        "sex": sex,
        "bmi": bmi.round(1),
        "alt": alt.round(1),
        "ast": ast.round(1),
        "alp": alp.round(1),
        "ggt": ggt.round(1),
        "bilirubin": bilirubin.round(2),
        "albumin": albumin.round(2),
        "total_protein": total_protein.round(2),
        "platelets": platelets.round(0).astype(int),
        "inr": inr.round(2),
        "diabetes": diabetes,
        "alcohol_units_per_week": alcohol_units_per_week.round(1),
        "family_history": family_history,
        "target": target,
    })


if __name__ == "__main__":
    kidney_df = generate_kidney()
    kidney_df.to_csv("kidney_data.csv", index=False)
    print(f"Wrote {len(kidney_df)} rows to kidney_data.csv | positive rate: {kidney_df['target'].mean():.2%}")

    liver_df = generate_liver()
    liver_df.to_csv("liver_data.csv", index=False)
    print(f"Wrote {len(liver_df)} rows to liver_data.csv | positive rate: {liver_df['target'].mean():.2%}")
