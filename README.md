# Kidney & Liver Disease Risk Estimator

A companion project to the Heart Health Risk Predictor, using the same
approach: synthetic clinical data, a trained risk model, and both a
desktop app and a browser app on top of it — this time covering **chronic
kidney disease** and **liver disease**.

**⚠️ Important:** Trained on **synthetic data** generated to mimic
commonly cited clinical risk factors (eGFR, creatinine, ACR, ALT/AST,
bilirubin, etc.) — not real patient data. Educational demo only, not a
medical device. Always consult a doctor.

## What's included

| File | Purpose |
|---|---|
| `generate_data.py` | Creates `kidney_data.csv` and `liver_data.csv`, 4,000 synthetic profiles each. |
| `train_model.py` | Trains Logistic Regression and Random Forest for both diseases, saves `kidney_model.joblib` / `liver_model.joblib`. |
| `predict.py` | CLI: `python predict.py --disease kidney --interactive` (or `liver`). |
| `organ_health_app.py` | **Desktop app** (Tkinter) — two tabs (Kidney / Liver), sliders for every lab value, live risk panel. |
| `organ_health_app.html` | **Browser app** — same idea, zero install, works by double-clicking the file. |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Recommended — desktop app, auto-trains on first run
python organ_health_app.py

# Or the browser version — just double-click organ_health_app.html

# Or the CLI
python generate_data.py
python train_model.py
python predict.py --disease kidney --interactive
python predict.py --disease liver --interactive
```

## Inputs used

**Kidney**: age, sex, systolic/diastolic blood pressure, creatinine, eGFR,
BUN, sodium, potassium, hemoglobin, urine ACR, HbA1c, diabetes,
hypertension, BMI, smoking, family history.

**Liver**: age, sex, BMI, ALT, AST, ALP, GGT, bilirubin, albumin, total
protein, platelets, INR, diabetes, alcohol use, family history.

## A note on model choice

Both models default to **logistic regression** rather than random forest,
even when the random forest scores a slightly higher ROC-AUC on the test
set. Random forests don't extrapolate past the range of their training
data — for an app where every input is a slider a user can push to
extreme values, that means a tree model's risk score can flatten out
instead of climbing for a genuinely severe profile. Logistic regression
stays monotonic and responsive across the whole slider range, which
matters more here than a marginal AUC gain. See `train_model.py` for the
exact selection rule.

## Swapping in real data

Replace `kidney_data.csv` / `liver_data.csv` with a real, ethically
sourced, properly licensed clinical dataset using the same column names
(or update the feature lists in `train_model.py`), then re-run
`train_model.py`. Any real deployment should be validated by medical
professionals before use beyond education.

## Publishing the browser app

Same process as the heart health app — `organ_health_app.html` is a
single self-contained file. Rename it to `index.html` and drop it into
GitHub Pages, Netlify, or Vercel to make it public. See the heart app's
publishing instructions for the full walkthrough.
