# Customer Churn Prediction Dashboard

End-to-end ML project that predicts customer churn for a telecom provider, explains
each prediction with SHAP, and serves results through an interactive Streamlit
dashboard.

> **Status:** Step 1 (data pipeline) complete. Steps 2–4 (training, explainability, dashboard) are next.

---

## Tech stack

| Layer            | Tool                                 |
| ---------------- | ------------------------------------ |
| Data             | pandas, scikit-learn, imbalanced-learn |
| Models           | Logistic Regression, XGBoost, LightGBM |
| Tuning           | Optuna                               |
| Tracking         | MLflow (local)                       |
| Explainability   | SHAP                                 |
| Dashboard        | Streamlit + Plotly                   |

---

## Project layout

```
churn-prediction/
├── data/
│   ├── raw/              # Original Telco Churn CSV (download below)
│   └── processed/        # Cleaned + split datasets
├── models/               # Trained .pkl artefacts
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_pipeline.py  # Step 1
│   ├── train.py          # Step 2
│   ├── evaluate.py       # Step 2
│   └── explain.py        # Step 3
├── app/
│   └── streamlit_app.py  # Step 4
├── reports/figures/      # SHAP plots, ROC curves, etc.
├── requirements.txt
└── README.md
```

---

## Setup

```powershell
# 1. Create + activate a virtualenv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt
```

---

## Step 1 — Data pipeline

### Download the dataset

The project uses the **IBM Telco Customer Churn** dataset (~7 K rows, free).

1. Sign in to Kaggle: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Save it to `data/raw/telco_churn.csv` (rename it).

### Run the pipeline

```powershell
python -m src.data_pipeline
```

This will:

1. Load and clean the raw CSV (fix `TotalCharges` blanks, drop ID).
2. Engineer features:
   - `tenure_group` ∈ {new, mid, loyal}
   - `charges_x_tenure` interaction
   - `has_multiple_services` bundled flag
3. One-hot encode categoricals.
4. Stratified split → train / val / test (60 / 20 / 20).
5. Print class imbalance and apply SMOTE on the **training set only**.
6. Save processed splits to `data/processed/` as versioned CSVs.

Outputs:

```
data/processed/
├── X_train.csv      # SMOTE-balanced features
├── y_train.csv
├── X_val.csv
├── y_val.csv
├── X_test.csv
├── y_test.csv
└── feature_columns.json   # Column order, used by the dashboard
```

---

## Roadmap

- [x] Step 1 — Data pipeline
- [ ] Step 2 — Train + tune three models with MLflow tracking
- [ ] Step 3 — SHAP explainability artefacts
- [ ] Step 4 — Streamlit multi-page dashboard
- [ ] Step 5 — Deploy to Streamlit Community Cloud
