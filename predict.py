"""
predict.py
----------
Command-line predictor for the Kidney & Liver Disease Risk models.
Loads kidney_model.joblib / liver_model.joblib and predicts a risk
probability for one person, either interactively or via flags.

Examples:
    python predict.py --disease kidney --interactive
    python predict.py --disease liver --interactive

    python predict.py --disease kidney --age 62 --sex 1 --systolic_bp 148 \\
        --diastolic_bp 92 --creatinine 1.6 --egfr 52 --bun 28 --sodium 138 \\
        --potassium 4.8 --hemoglobin 11.5 --urine_acr 45 --hba1c 7.2 \\
        --diabetes 1 --hypertension 1 --bmi 31 --smoker 1 --family_history 0

DISCLAIMER: This tool uses models trained on SYNTHETIC data and is for
educational / demonstration purposes only. It is NOT a medical device and
must never be used to make real health decisions. Always consult a
qualified healthcare professional.
"""

import argparse

import joblib
import pandas as pd

KIDNEY_HELP = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "systolic_bp": "Systolic blood pressure (mmHg)",
    "diastolic_bp": "Diastolic blood pressure (mmHg)",
    "creatinine": "Serum creatinine (mg/dL)",
    "egfr": "eGFR (mL/min/1.73m^2)",
    "bun": "Blood urea nitrogen (mg/dL)",
    "sodium": "Sodium (mmol/L)",
    "potassium": "Potassium (mmol/L)",
    "hemoglobin": "Hemoglobin (g/dL)",
    "urine_acr": "Urine albumin-creatinine ratio (mg/g)",
    "hba1c": "HbA1c (%)",
    "diabetes": "Diagnosed diabetes? (1 = yes, 0 = no)",
    "hypertension": "Diagnosed hypertension? (1 = yes, 0 = no)",
    "bmi": "Body mass index",
    "smoker": "Current smoker? (1 = yes, 0 = no)",
    "family_history": "Family history of kidney disease? (1 = yes, 0 = no)",
}

LIVER_HELP = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "bmi": "Body mass index",
    "alt": "ALT (U/L)",
    "ast": "AST (U/L)",
    "alp": "ALP (U/L)",
    "ggt": "GGT (U/L)",
    "bilirubin": "Total bilirubin (mg/dL)",
    "albumin": "Albumin (g/dL)",
    "total_protein": "Total protein (g/dL)",
    "platelets": "Platelet count (x10^9/L)",
    "inr": "INR",
    "diabetes": "Diagnosed diabetes? (1 = yes, 0 = no)",
    "alcohol_units_per_week": "Alcohol units per week",
    "family_history": "Family history of liver disease? (1 = yes, 0 = no)",
}

DISEASES = {
    "kidney": {"model": "kidney_model.joblib", "help": KIDNEY_HELP},
    "liver": {"model": "liver_model.joblib", "help": LIVER_HELP},
}


def prompt_for_inputs(field_help):
    print("Answer the following questions (numbers only).\n")
    answers = {}
    for field, help_text in field_help.items():
        while True:
            raw = input(f"{help_text}\n  {field} = ").strip()
            try:
                answers[field] = float(raw)
                break
            except ValueError:
                print("  Please enter a numeric value.")
    return answers


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Predict kidney or liver disease risk.")
    parser.add_argument("--disease", choices=["kidney", "liver"], required=True)
    parser.add_argument("--interactive", action="store_true")
    # register both field sets; only the relevant ones are required at runtime
    all_fields = {**KIDNEY_HELP, **LIVER_HELP}
    for field, help_text in all_fields.items():
        if not any(a.dest == field for a in parser._actions):
            parser.add_argument(f"--{field}", type=float, help=help_text)
    return parser


def risk_bucket(prob: float) -> str:
    if prob < 0.2:
        return "Low"
    if prob < 0.5:
        return "Moderate"
    if prob < 0.75:
        return "High"
    return "Very High"


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = DISEASES[args.disease]
    field_help = cfg["help"]
    bundle = joblib.load(cfg["model"])
    pipeline, features = bundle["pipeline"], bundle["features"]

    if args.interactive or not any(getattr(args, f, None) is not None for f in field_help):
        values = prompt_for_inputs(field_help)
    else:
        missing = [f for f in field_help if getattr(args, f, None) is None]
        if missing:
            parser.error(f"Missing required fields: {', '.join(missing)} (or use --interactive)")
        values = {f: getattr(args, f) for f in field_help}

    row = pd.DataFrame([{f: values[f] for f in features}])
    prob = pipeline.predict_proba(row)[0, 1]
    bucket = risk_bucket(prob)

    print("\n" + "=" * 44)
    print(f" {cfg is cfg and args.disease.capitalize()} disease — estimated risk: {prob:.1%}")
    print(f" Risk category: {bucket}")
    print("=" * 44)
    print(
        "\nThis estimate comes from a model trained on SYNTHETIC data and is "
        "for educational purposes only. It is not a diagnosis. Please talk "
        "to a doctor about your real health, especially if this result "
        "concerns you."
    )


if __name__ == "__main__":
    main()
