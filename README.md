# Customer Churn Prediction Dashboard

End-to-end ML project that predicts customer churn for a telecom provider, explains
each prediction with SHAP, and serves results through an interactive Streamlit
dashboard.

## Tech stack

| Layer            | Tool                                   |
| ---------------- | -------------------------------------- |
| Data             | pandas, scikit-learn, imbalanced-learn |
| Models           | Logistic Regression, XGBoost, LightGBM |
| Tuning           | Optuna                                 |
| Tracking         | MLflow (SQLite backend)                |
| Explainability   | SHAP                                   |
| Dashboard        | Streamlit + Plotly                     |

## Project structure

```
churn-prediction/
├── data/
│   ├── raw/              # IBM Telco Customer Churn CSV
│   └── processed/        # Cleaned + split datasets
├── models/               # Trained .pkl artefacts + metadata
├── src/
│   ├── data_pipeline.py  # Load, clean, engineer, split, SMOTE
│   ├── train.py          # Optuna tuning + MLflow tracking, 3 model families
│   ├── evaluate.py       # Metric + plot utilities
│   └── explain.py        # SHAP explainability (TreeSHAP / LinearSHAP)
├── app/
│   └── streamlit_app.py  # Multi-page dashboard
├── reports/figures/      # ROC, PR, confusion matrix, SHAP plots
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the **IBM Telco Customer Churn** dataset from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save
the CSV to `data/raw/telco_churn.csv`.

## Usage

```powershell
python -m src.data_pipeline     # build train/val/test + SMOTE-balanced training set
python -m src.train             # tune LR / XGBoost / LightGBM with Optuna + MLflow
python -m src.explain           # generate SHAP plots and dashboard cache
streamlit run app/streamlit_app.py
```

### MLflow UI (optional)

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open <http://localhost:5000>.
