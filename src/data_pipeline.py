"""
Step 1 — Data pipeline for the IBM Telco Customer Churn dataset.

Run from the project root:

    python -m src.data_pipeline

Inputs
------
data/raw/telco_churn.csv   (download from Kaggle, see README)

Outputs
-------
data/processed/X_train.csv, y_train.csv     (SMOTE-balanced)
data/processed/X_val.csv,   y_val.csv       (untouched, real distribution)
data/processed/X_test.csv,  y_test.csv      (untouched, real distribution)
data/processed/feature_columns.json         (column order, used by the dashboard)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Paths + config
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.20         # 20% held out as test
VAL_SIZE = 0.25          # 25% of the remaining 80% => 20% of total
TARGET_COL = "Churn"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_pipeline")


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


# --------------------------------------------------------------------------- #
# Load + clean
# --------------------------------------------------------------------------- #

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download it from Kaggle (see README) and save as "
            "data/raw/telco_churn.csv."
        )
    df = pd.read_csv(path)
    log.info("Loaded raw data: %s rows, %s columns", *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known issues in the Telco CSV."""
    df = df.copy()

    # Drop the customer ID — it's a row identifier, not a feature.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # `TotalCharges` ships as object because new customers have a blank string
    # instead of a number. Coerce to float and impute the (very few) NaNs with 0
    # — these are customers with tenure == 0, who haven't been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = df["TotalCharges"].isna().sum()
    if n_blank:
        log.info("Imputing %d blank TotalCharges (tenure==0 customers) with 0", n_blank)
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Target → 0/1
    df[TARGET_COL] = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)

    # `SeniorCitizen` is already 0/1 but typed as int; keep it numeric.
    return df


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

# Service columns used for the "bundled services" flag.
_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def _tenure_bucket(t: int) -> str:
    if t <= 12:
        return "new"
    if t <= 48:
        return "mid"
    return "loyal"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Tenure groups: new (<=1y) / mid (1–4y) / loyal (>4y)
    df["tenure_group"] = df["tenure"].apply(_tenure_bucket).astype("category")

    # 2. Spend × loyalty interaction — captures "high-value long-tenure" cohort.
    df["charges_x_tenure"] = df["MonthlyCharges"] * df["tenure"]

    # 3. Bundled-services flag: does the customer have 2+ active services?
    #    A service is "active" when the value is neither "No" nor a no-internet
    #    sentinel like "No internet service" / "No phone service".
    def _has_service(val: object) -> bool:
        s = str(val).strip().lower()
        return s not in {"no", "no internet service", "no phone service"}

    service_active = df[_SERVICE_COLS].map(_has_service)
    df["has_multiple_services"] = (service_active.sum(axis=1) >= 2).astype(int)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all object/category columns except the target."""
    cat_cols = [
        c for c in df.columns
        if (df[c].dtype == "object" or str(df[c].dtype) == "category") and c != TARGET_COL
    ]
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # get_dummies produces bool columns on pandas 2.x — cast to int8 for downstream models.
    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype(np.int8)
    return encoded


# --------------------------------------------------------------------------- #
# Split + SMOTE
# --------------------------------------------------------------------------- #

def split(df: pd.DataFrame) -> Splits:
    """Stratified 60 / 20 / 20 split on the churn label."""
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE, stratify=y_trainval, random_state=RANDOM_STATE,
    )

    log.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)


def apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Balance the training set with SMOTE. Validation/test are left untouched."""
    before = y.value_counts().to_dict()
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X, y)
    after = pd.Series(y_res).value_counts().to_dict()
    log.info("Class balance — before SMOTE: %s | after: %s", before, after)
    # fit_resample returns numpy arrays in some sklearn versions; keep as DataFrame.
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=TARGET_COL)
    return X_res, y_res


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_splits(splits: Splits, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    splits.X_train.to_csv(out_dir / "X_train.csv", index=False)
    splits.y_train.to_csv(out_dir / "y_train.csv", index=False)
    splits.X_val.to_csv(out_dir / "X_val.csv", index=False)
    splits.y_val.to_csv(out_dir / "y_val.csv", index=False)
    splits.X_test.to_csv(out_dir / "X_test.csv", index=False)
    splits.y_test.to_csv(out_dir / "y_test.csv", index=False)

    # Persist column order so the dashboard / batch scorer can align inputs.
    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(list(splits.X_train.columns), f, indent=2)

    log.info("Wrote processed splits to %s", out_dir)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run() -> Splits:
    df = load_raw()
    df = clean(df)
    log.info("Class distribution (raw): %s", df[TARGET_COL].value_counts().to_dict())

    df = engineer_features(df)
    df = encode(df)
    log.info("Encoded feature matrix shape: %s", df.shape)

    splits = split(df)
    X_train_bal, y_train_bal = apply_smote(splits.X_train, splits.y_train)
    splits = Splits(
        X_train=X_train_bal, y_train=y_train_bal,
        X_val=splits.X_val, y_val=splits.y_val,
        X_test=splits.X_test, y_test=splits.y_test,
    )
    save_splits(splits)
    log.info("Step 1 complete.")
    return splits


if __name__ == "__main__":
    run()
