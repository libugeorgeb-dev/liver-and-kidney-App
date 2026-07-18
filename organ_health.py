"""
organ_health_app.py
--------------------
A standalone desktop app (Tkinter, ships with Python) for the Kidney &
Liver Disease Risk Predictor. Two tabs, one per organ, each with its own
sliders/dropdowns and live risk panel.

Run it with:
    python organ_health_app.py

On first run it auto-generates the synthetic training data and trains both
models if they don't exist yet — this file works standalone.

DISCLAIMER: Educational demo only, trained on SYNTHETIC data. Not a
medical device. Not a diagnosis. Always consult a real doctor.
"""

import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def ensure_ready(status_callback=None):
    if not os.path.exists(os.path.join(HERE, "kidney_data.csv")):
        if status_callback:
            status_callback("Generating training data...")
        subprocess.run([sys.executable, os.path.join(HERE, "generate_data.py")], cwd=HERE, check=True)
    if not (os.path.exists(os.path.join(HERE, "kidney_model.joblib"))
            and os.path.exists(os.path.join(HERE, "liver_model.joblib"))):
        if status_callback:
            status_callback("Training models (first run only)...")
        subprocess.run([sys.executable, os.path.join(HERE, "train_model.py")], cwd=HERE, check=True)


# ---------------------------------------------------------------- styling --
BG = "#EEF1F6"
PANEL = "#FFFFFF"
INK = "#152238"
INK_SOFT = "#4B5A72"
TEAL = "#0B6E4F"
TEAL_SOFT = "#E4F2EC"
GOLD = "#8A6A15"
GOLD_SOFT = "#FAF1DD"
RED = "#A5313C"
RED_SOFT = "#FBE7E9"
DEEP = "#0D1B2E"

FONT_H1 = ("Georgia", 20, "bold")
FONT_H2 = ("Georgia", 13, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 10)
FONT_RISK = ("Georgia", 34, "bold")


class OrganPanel(tk.Frame):
    """One organ's full input + result UI, embedded inside a notebook tab."""

    def __init__(self, parent, app, organ_key, sections, model_bundle):
        super().__init__(parent, bg=BG)
        self.app = app
        self.organ_key = organ_key
        self.bundle = model_bundle
        self.vars = {}
        self._dropdown_maps = {}

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=(16, 4))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left_container = tk.Frame(body, bg=BG)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        canvas = tk.Canvas(left_container, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self._build_inputs(scroll_frame, sections)

        right = tk.Frame(body, bg=DEEP, padx=20, pady=20)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_result_panel(right)

        self._calculate()

    # ------------------------------------------------------------ inputs --
    def _section(self, parent, num, title, sub):
        card = tk.Frame(parent, bg=PANEL, highlightbackground="#D7DCE6", highlightthickness=1)
        card.pack(fill="x", pady=(0, 14), ipady=6)
        head = tk.Frame(card, bg=PANEL)
        head.pack(fill="x", padx=18, pady=(14, 2))
        tk.Label(head, text=num, font=FONT_MONO, bg=PANEL, fg=TEAL).pack(side="left", padx=(0, 8))
        tk.Label(head, text=title, font=FONT_H2, bg=PANEL, fg=INK).pack(side="left")
        tk.Label(card, text=sub, font=FONT_SMALL, bg=PANEL, fg=INK_SOFT).pack(anchor="w", padx=18, pady=(0, 8))
        inner = tk.Frame(card, bg=PANEL)
        inner.pack(fill="x", padx=18, pady=(0, 6))
        return inner

    def _slider(self, parent, key, label, lo, hi, default, resolution=1, unit=""):
        var = tk.DoubleVar(value=default)
        self.vars[key] = var
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=6)
        top = tk.Frame(row, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text=label, font=FONT_BODY, bg=PANEL, fg=INK).pack(side="left")
        val_lbl = tk.Label(top, text=f"{default}{unit}", font=FONT_MONO, bg=PANEL, fg=TEAL)
        val_lbl.pack(side="right")

        def on_change(_evt=None):
            v = var.get()
            v = round(v, 1) if resolution < 1 else int(v)
            val_lbl.config(text=f"{v}{unit}")
            self._calculate_debounced()

        scale = tk.Scale(
            row, from_=lo, to=hi, orient="horizontal", variable=var, resolution=resolution,
            showvalue=False, bg=PANEL, troughcolor="#D7DCE6", highlightthickness=0,
            fg=TEAL, activebackground=TEAL, sliderrelief="flat", command=lambda e: on_change(),
        )
        scale.pack(fill="x")

    def _dropdown(self, parent, key, label, options, default_index=0):
        var = tk.StringVar(value=options[default_index][0])
        self.vars[key] = var
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=6)
        tk.Label(row, text=label, font=FONT_BODY, bg=PANEL, fg=INK).pack(anchor="w")
        combo = ttk.Combobox(row, values=[o[0] for o in options], state="readonly", font=FONT_BODY)
        combo.current(default_index)
        combo.pack(fill="x", pady=(4, 0))
        combo.bind("<<ComboboxSelected>>", lambda e: self._calculate_debounced())
        self._dropdown_maps[key] = {o[0]: o[1] for o in options}

    def _toggle(self, parent, key, label, default=False):
        var = tk.BooleanVar(value=default)
        self.vars[key] = var
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, font=FONT_BODY, bg=PANEL, fg=INK).pack(side="left")
        tk.Checkbutton(row, variable=var, bg=PANEL, activebackground=PANEL,
                        command=self._calculate_debounced).pack(side="right")

    def _build_inputs(self, parent, sections):
        for num, title, sub, fields in sections:
            s = self._section(parent, num, title, sub)
            for f in fields:
                kind = f[0]
                if kind == "slider":
                    _, key, label, lo, hi, default, res, unit = f
                    self._slider(s, key, label, lo, hi, default, res, unit)
                elif kind == "dropdown":
                    _, key, label, options, default_idx = f
                    self._dropdown(s, key, label, options, default_idx)
                elif kind == "toggle":
                    _, key, label, default = f
                    self._toggle(s, key, label, default)

        btn = tk.Button(
            parent, text="Calculate Risk", font=("Segoe UI", 11, "bold"),
            bg=TEAL, fg="white", activebackground="#0a5c42", relief="flat",
            padx=16, pady=10, command=self._calculate, cursor="hand2",
        )
        btn.pack(fill="x", pady=(6, 20))

    # ------------------------------------------------------------ result --
    def _build_result_panel(self, right):
        tk.Label(right, text="ESTIMATED RISK", font=("Consolas", 10), bg=DEEP, fg="#7FA99B").pack(anchor="w")
        self.risk_number = tk.Label(right, text="—%", font=FONT_RISK, bg=DEEP, fg="white")
        self.risk_number.pack(anchor="w", pady=(4, 4))
        self.risk_badge = tk.Label(right, text="Calculating...", font=("Segoe UI", 10, "bold"),
                                    bg=TEAL_SOFT, fg=TEAL, padx=12, pady=4)
        self.risk_badge.pack(anchor="w", pady=(0, 16))
        self.risk_desc = tk.Label(right, text="", font=FONT_SMALL, bg=DEEP, fg="#C3CBDA",
                                   wraplength=280, justify="left")
        self.risk_desc.pack(anchor="w", pady=(0, 18))
        tk.Frame(right, bg="#2a3a52", height=1).pack(fill="x", pady=(0, 14))
        tk.Label(right, text="TOP CONTRIBUTING FACTORS", font=("Consolas", 9), bg=DEEP, fg="#8FA0BD").pack(
            anchor="w", pady=(0, 8))
        self.factor_frame = tk.Frame(right, bg=DEEP)
        self.factor_frame.pack(fill="x")

    def _get_raw_values(self):
        raw = {}
        for key, var in self.vars.items():
            if isinstance(var, tk.BooleanVar):
                raw[key] = 1 if var.get() else 0
            elif isinstance(var, tk.StringVar):
                raw[key] = self._dropdown_maps[key][var.get()]
            else:
                raw[key] = var.get()
        return raw

    def _calculate_debounced(self):
        if hasattr(self, "_debounce_job"):
            self.after_cancel(self._debounce_job)
        self._debounce_job = self.after(200, self._calculate)

    def _calculate(self):
        raw = self._get_raw_values()
        pipeline, features = self.bundle["pipeline"], self.bundle["features"]
        row = pd.DataFrame([{f: raw[f] for f in features}])
        prob = pipeline.predict_proba(row)[0, 1]

        self.risk_number.config(text=f"{prob*100:.1f}%")
        if prob < 0.20:
            name, bg, fg = "Low risk", TEAL_SOFT, TEAL
            desc = "Few active risk factors present. Keep up healthy habits and routine checkups."
        elif prob < 0.50:
            name, bg, fg = "Moderate risk", GOLD_SOFT, GOLD
            desc = "A few factors are pushing risk upward. Worth discussing with a doctor at your next visit."
        elif prob < 0.75:
            name, bg, fg = "High risk", RED_SOFT, RED
            desc = "Several risk factors are compounding. A clinical evaluation is strongly advised."
        else:
            name, bg, fg = "Very high risk", "#F6D9DC", "#8F1F2A"
            desc = "Multiple significant risk factors present. Please seek a professional medical evaluation."
        self.risk_badge.config(text=name, bg=bg, fg=fg)
        self.risk_desc.config(text=desc)

        for w in self.factor_frame.winfo_children():
            w.destroy()
        contributions = self._estimate_contributions(raw)
        for label, val in contributions[:4]:
            row_f = tk.Frame(self.factor_frame, bg=DEEP)
            row_f.pack(fill="x", pady=3)
            tk.Label(row_f, text=label, font=FONT_SMALL, bg=DEEP, fg="#DBE1EC").pack(side="left")
            tk.Label(row_f, text=f"+{val:.2f}", font=("Consolas", 9, "bold"), bg=DEEP, fg="white").pack(side="right")

    def _estimate_contributions(self, v):
        if self.organ_key == "kidney":
            contribs = [
                ("Age", 0.035 * v["age"]),
                ("Low eGFR", 0.9 * (1 if v["egfr"] < 60 else 0)),
                ("High creatinine", 0.7 * (1 if v["creatinine"] > 1.3 else 0)),
                ("Elevated urine ACR", 0.8 * (1 if v["urine_acr"] >= 30 else 0)),
                ("High systolic BP", 0.5 * (1 if v["systolic_bp"] > 140 else 0)),
                ("High diastolic BP", 0.4 * (1 if v["diastolic_bp"] > 90 else 0)),
                ("Elevated HbA1c", 0.6 * (1 if v["hba1c"] >= 6.5 else 0)),
                ("Age over 60", 0.35 * (1 if v["age"] > 60 else 0)),
                ("Smoking", 0.3 * v["smoker"]),
                ("Family history", 0.35 * v["family_history"]),
            ]
        else:
            contribs = [
                ("Age", 0.02 * v["age"]),
                ("Elevated ALT", 0.6 * (1 if v["alt"] > 40 else 0)),
                ("Elevated AST", 0.6 * (1 if v["ast"] > 40 else 0)),
                ("Elevated bilirubin", 0.7 * (1 if v["bilirubin"] > 1.2 else 0)),
                ("Low albumin", 0.6 * (1 if v["albumin"] < 3.5 else 0)),
                ("High INR", 0.5 * (1 if v["inr"] > 1.2 else 0)),
                ("BMI over 30", 0.5 * (1 if v["bmi"] > 30 else 0)),
                ("Diabetes", 0.45 * v["diabetes"]),
                ("Heavy alcohol use", 0.55 * (1 if v["alcohol_units_per_week"] > 21 else 0)),
                ("Family history", 0.3 * v["family_history"]),
            ]
        contribs = [c for c in contribs if c[1] > 0.02]
        contribs.sort(key=lambda c: c[1], reverse=True)
        return contribs


class OrganHealthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Organ Health Check — Kidney & Liver Risk Estimator")
        self.geometry("940x700")
        self.configure(bg=BG)
        self.minsize(860, 640)

        self.loading_frame = tk.Frame(self, bg=BG)
        self.loading_frame.pack(fill="both", expand=True)
        tk.Label(self.loading_frame, text="Organ Health Check", font=FONT_H1, bg=BG, fg=INK).pack(pady=(230, 6))
        self.status_label = tk.Label(self.loading_frame, text="Starting up...", font=FONT_BODY, bg=BG, fg=INK_SOFT)
        self.status_label.pack()

        self.after(150, self._load_models)

    def _set_status(self, text):
        self.status_label.config(text=text)
        self.update_idletasks()

    def _load_models(self):
        try:
            ensure_ready(self._set_status)
            self._set_status("Loading models...")
            self.kidney_bundle = joblib.load(os.path.join(HERE, "kidney_model.joblib"))
            self.liver_bundle = joblib.load(os.path.join(HERE, "liver_model.joblib"))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Error: {exc}")
            return
        self.loading_frame.destroy()
        self._build_main_ui()

    def _build_main_ui(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 6))
        tk.Label(header, text="Organ Health Check", font=FONT_H1, bg=BG, fg=INK).pack(anchor="w")
        tk.Label(header, text="Choose a tab, adjust the profile, and get a live risk estimate.",
                 font=FONT_SMALL, bg=BG, fg=INK_SOFT).pack(anchor="w")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[16, 8])

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=14, pady=(6, 0))

        kidney_tab = OrganPanel(notebook, self, "kidney", KIDNEY_SECTIONS, self.kidney_bundle)
        liver_tab = OrganPanel(notebook, self, "liver", LIVER_SECTIONS, self.liver_bundle)
        notebook.add(kidney_tab, text="  Kidney  ")
        notebook.add(liver_tab, text="  Liver  ")

        tk.Label(
            self,
            text=("Not a medical device. Trained on synthetic data for educational use only. "
                  "This is not a diagnosis — please consult a licensed healthcare professional."),
            font=FONT_SMALL, bg=TEAL_SOFT, fg=TEAL, wraplength=880, justify="left", padx=14, pady=10,
        ).pack(fill="x", padx=24, pady=(10, 16))


# ------------------------------------------------------------ field specs --
KIDNEY_SECTIONS = [
    ("01", "Demographics", "Basic profile", [
        ("slider", "age", "Age", 18, 90, 52, 1, ""),
        ("dropdown", "sex", "Sex", [("Male", 1), ("Female", 0)], 0),
        ("slider", "bmi", "BMI", 15, 50, 26, 0.1, ""),
    ]),
    ("02", "Blood pressure & sugar", "Hypertension and diabetes indicators", [
        ("slider", "systolic_bp", "Systolic blood pressure", 90, 210, 130, 1, " mmHg"),
        ("slider", "diastolic_bp", "Diastolic blood pressure", 55, 130, 80, 1, " mmHg"),
        ("slider", "hba1c", "HbA1c", 4.0, 13.0, 5.4, 0.1, "%"),
        ("toggle", "diabetes", "Diagnosed diabetes", False),
        ("toggle", "hypertension", "Diagnosed hypertension", False),
    ]),
    ("03", "Kidney function labs", "Core markers of kidney health", [
        ("slider", "egfr", "eGFR", 5, 130, 90, 1, " mL/min"),
        ("slider", "creatinine", "Creatinine", 0.4, 8.0, 0.9, 0.1, " mg/dL"),
        ("slider", "bun", "Blood urea nitrogen (BUN)", 5, 100, 14, 1, " mg/dL"),
        ("slider", "urine_acr", "Urine albumin-creatinine ratio (ACR)", 1, 300, 10, 1, " mg/g"),
    ]),
    ("04", "Other labs & history", "Electrolytes, blood count, lifestyle", [
        ("slider", "sodium", "Sodium", 125, 152, 140, 1, " mmol/L"),
        ("slider", "potassium", "Potassium", 2.8, 7.5, 4.2, 0.1, " mmol/L"),
        ("slider", "hemoglobin", "Hemoglobin", 6, 18, 14, 0.1, " g/dL"),
        ("toggle", "smoker", "Current smoker", False),
        ("toggle", "family_history", "Family history of kidney disease", False),
    ]),
]

LIVER_SECTIONS = [
    ("01", "Demographics", "Basic profile", [
        ("slider", "age", "Age", 18, 90, 48, 1, ""),
        ("dropdown", "sex", "Sex", [("Male", 1), ("Female", 0)], 0),
        ("slider", "bmi", "BMI", 15, 50, 26, 0.1, ""),
    ]),
    ("02", "Liver enzymes", "ALT, AST, ALP, GGT", [
        ("slider", "alt", "ALT", 5, 400, 25, 1, " U/L"),
        ("slider", "ast", "AST", 5, 400, 25, 1, " U/L"),
        ("slider", "alp", "ALP", 30, 400, 80, 1, " U/L"),
        ("slider", "ggt", "GGT", 5, 500, 30, 1, " U/L"),
    ]),
    ("03", "Liver function labs", "Synthesis and clearance markers", [
        ("slider", "bilirubin", "Bilirubin", 0.1, 15.0, 0.8, 0.1, " mg/dL"),
        ("slider", "albumin", "Albumin", 1.5, 5.5, 4.2, 0.1, " g/dL"),
        ("slider", "total_protein", "Total protein", 4.0, 9.5, 7.1, 0.1, " g/dL"),
        ("slider", "platelets", "Platelets", 30, 500, 250, 1, " x10^9/L"),
        ("slider", "inr", "INR", 0.8, 4.0, 1.0, 0.1, ""),
    ]),
    ("04", "Lifestyle & history", "Alcohol, diabetes, family history", [
        ("slider", "alcohol_units_per_week", "Alcohol units per week", 0, 90, 4, 1, ""),
        ("toggle", "diabetes", "Diagnosed diabetes", False),
        ("toggle", "family_history", "Family history of liver disease", False),
    ]),
]


if __name__ == "__main__":
    app = OrganHealthApp()
    app.mainloop()
