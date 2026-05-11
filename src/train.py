"""
Step 2 — Train and tune three models, track every experiment, save the winner.

Models
------
1. Logistic Regression  (linear baseline; standardised features)
2. XGBoost              (gradient boosting, depth-wise)
3. LightGBM             (gradient boosting, leaf-wise)

Pipeline per model:
  1. Optuna runs N_TRIALS hyperparameter trials (TPE sampler, seed=42).
  2. Each trial logs its params + validation AUC to MLflow as a nested run.
  3. The best params are refit on the SMOTE-balanced training set.
  4. The fitted model is evaluated on the held-out test set.

The winner (highest validation AUC) is saved to models/best_model.pkl with
joblib, alongside a JSON metadata file describing every run.

Run from the project root:

    python -m src.train

View MLflow UI afterwards:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
    # then open http://localhost:5000
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate_model

# --------------------------------------------------------------------------- #
# Paths + config
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

# MLflow: SQLite backend (file-based backend doesn't support the Traces UI in 2.13+).
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

N_TRIALS = 20
RANDOM_STATE = 42
EXPERIMENT_NAME = "churn-prediction"

# Quieten noisy libraries — Optuna prints every trial otherwise.
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

@dataclass
class Data:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val:   pd.DataFrame
    y_val:   pd.Series
    X_test:  pd.DataFrame
    y_test:  pd.Series


def load_data() -> Data:
    def _read_y(name: str) -> pd.Series:
        return pd.read_csv(PROCESSED_DIR / name).squeeze("columns")

    return Data(
        X_train=pd.read_csv(PROCESSED_DIR / "X_train.csv"),
        y_train=_read_y("y_train.csv"),
        X_val=pd.read_csv(PROCESSED_DIR / "X_val.csv"),
        y_val=_read_y("y_val.csv"),
        X_test=pd.read_csv(PROCESSED_DIR / "X_test.csv"),
        y_test=_read_y("y_test.csv"),
    )


# --------------------------------------------------------------------------- #
# Per-model tuners
# --------------------------------------------------------------------------- #

def _log_trial(run_name: str, params: dict, auc: float) -> None:
    """Log one Optuna trial as a nested MLflow run."""
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_auc", auc)


def tune_logistic_regression(data: Data) -> tuple[Pipeline, dict]:
    log.info("Tuning Logistic Regression — %d trials", N_TRIALS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "C":         trial.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty":   trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver":    "liblinear",          # fast on small data, supports l1+l2
            "max_iter":  1000,
            "random_state": RANDOM_STATE,
        }
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ])
        pipe.fit(data.X_train, data.y_train)
        proba = pipe.predict_proba(data.X_val)[:, 1]
        auc = roc_auc_score(data.y_val, proba)
        _log_trial(f"logreg_trial_{trial.number:02d}", {**params, "model": "logreg"}, auc)
        return auc

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = {
        **study.best_params,
        "solver":    "liblinear",
        "max_iter":  1000,
        "random_state": RANDOM_STATE,
    }
    best_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**best)),
    ])
    best_pipe.fit(data.X_train, data.y_train)

    log.info("  → best val AUC: %.4f  params: %s", study.best_value, study.best_params)
    return best_pipe, {"best_val_auc": float(study.best_value), "best_params": study.best_params}


def tune_xgboost(data: Data) -> tuple[xgb.XGBClassifier, dict]:
    log.info("Tuning XGBoost — %d trials", N_TRIALS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "random_state":     RANDOM_STATE,
            "eval_metric":      "logloss",
            "n_jobs":           -1,
            "verbosity":        0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(data.X_train, data.y_train)
        proba = model.predict_proba(data.X_val)[:, 1]
        auc = roc_auc_score(data.y_val, proba)
        _log_trial(f"xgb_trial_{trial.number:02d}", {**params, "model": "xgboost"}, auc)
        return auc

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = {
        **study.best_params,
        "random_state": RANDOM_STATE,
        "eval_metric":  "logloss",
        "n_jobs":       -1,
        "verbosity":    0,
    }
    best_model = xgb.XGBClassifier(**best)
    best_model.fit(data.X_train, data.y_train)

    log.info("  → best val AUC: %.4f  params: %s", study.best_value, study.best_params)
    return best_model, {"best_val_auc": float(study.best_value), "best_params": study.best_params}


def tune_lightgbm(data: Data) -> tuple[lgb.LGBMClassifier, dict]:
    log.info("Tuning LightGBM — %d trials", N_TRIALS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(data.X_train, data.y_train)
        proba = model.predict_proba(data.X_val)[:, 1]
        auc = roc_auc_score(data.y_val, proba)
        _log_trial(f"lgbm_trial_{trial.number:02d}", {**params, "model": "lightgbm"}, auc)
        return auc

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = {
        **study.best_params,
        "random_state": RANDOM_STATE,
        "n_jobs":       -1,
        "verbose":      -1,
    }
    best_model = lgb.LGBMClassifier(**best)
    best_model.fit(data.X_train, data.y_train)

    log.info("  → best val AUC: %.4f  params: %s", study.best_value, study.best_params)
    return best_model, {"best_val_auc": float(study.best_value), "best_params": study.best_params}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # SQLite-backed MLflow tracking (gitignored). Artifacts go to ./mlartifacts/.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_data()
    log.info(
        "Loaded data — X_train: %s | X_val: %s | X_test: %s",
        data.X_train.shape, data.X_val.shape, data.X_test.shape,
    )

    results: dict[str, tuple[object, dict]] = {}

    # Each model family gets a parent run that groups its 20 trial runs.
    with mlflow.start_run(run_name="logreg_study"):
        results["logreg"] = tune_logistic_regression(data)

    with mlflow.start_run(run_name="xgboost_study"):
        results["xgboost"] = tune_xgboost(data)

    with mlflow.start_run(run_name="lightgbm_study"):
        results["lightgbm"] = tune_lightgbm(data)

    # ----- Validation summary ----- #
    log.info("=" * 60)
    log.info("Validation AUC comparison:")
    val_aucs = {name: meta["best_val_auc"] for name, (_, meta) in results.items()}
    for name, auc in sorted(val_aucs.items(), key=lambda kv: -kv[1]):
        log.info("  %-10s  AUC = %.4f", name, auc)

    best_name = max(val_aucs, key=val_aucs.get)
    best_model, best_meta = results[best_name]
    log.info("Winner by validation AUC: %s (%.4f)", best_name, val_aucs[best_name])

    # ----- Test-set evaluation for ALL three (dashboard will show comparisons) ----- #
    log.info("Evaluating all models on the held-out TEST set …")
    all_test_metrics: dict[str, dict] = {}
    for name, (model, _) in results.items():
        all_test_metrics[name] = evaluate_model(
            model, data.X_test, data.y_test, name=name, fig_dir=FIG_DIR,
        )
        log.info(
            "  %-10s  AUC=%.4f  F1=%.4f  P=%.4f  R=%.4f",
            name,
            all_test_metrics[name]["auc"],
            all_test_metrics[name]["f1"],
            all_test_metrics[name]["precision"],
            all_test_metrics[name]["recall"],
        )

    test_metrics = all_test_metrics[best_name]

    # ----- Persist winner ----- #
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)
    log.info("Saved winner to %s", model_path)

    metadata = {
        "best_model":        best_name,
        "best_val_auc":      val_aucs[best_name],
        "best_params":       best_meta["best_params"],
        "test_metrics":      test_metrics,
        "validation_aucs":   val_aucs,
        "all_test_metrics":  all_test_metrics,
        "n_trials_per_model": N_TRIALS,
        "random_state":      RANDOM_STATE,
    }
    metadata_path = MODELS_DIR / "best_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Saved metadata to %s", metadata_path)

    # ----- Top-level MLflow run with final artefacts ----- #
    with mlflow.start_run(run_name=f"BEST_{best_name}"):
        mlflow.log_param("winner_model", best_name)
        for k, v in best_meta["best_params"].items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("val_auc", val_aucs[best_name])
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metadata_path))

    log.info("Step 2 complete.")


if __name__ == "__main__":
    main()
