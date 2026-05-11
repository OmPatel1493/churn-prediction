"""
Step 3 — SHAP explainability for the winning model.

Loads the winner from models/best_model.pkl, computes SHAP values on the test
set, and saves three diagnostic plots plus a pickled SHAP cache the dashboard
will reuse.

Run from the project root:

    python -m src.explain
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# Paths + config
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

TOP_N_FEATURES = 15
WATERFALL_TARGET_PROBA = 0.85   # pick a confident-but-not-extreme churner
BG_SAMPLE_SIZE = 100            # background sample for LinearExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("explain")


# --------------------------------------------------------------------------- #
# Artefact loading
# --------------------------------------------------------------------------- #

def load_artefacts() -> tuple[object, dict, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the winning model, metadata, and the train/test feature matrices."""
    model_path = MODELS_DIR / "best_model.pkl"
    meta_path = MODELS_DIR / "best_metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Step 2 outputs missing. Run `python -m src.train` first to "
            f"create {model_path} and {meta_path}."
        )

    model = joblib.load(model_path)
    with meta_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return model, metadata, X_train, X_test, y_test


# --------------------------------------------------------------------------- #
# SHAP computation
# --------------------------------------------------------------------------- #

def compute_shap_values(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> shap.Explanation:
    """Pick the right SHAP explainer for the model type.

    - Tree models (XGBoost, LightGBM)  → TreeExplainer (exact, fast)
    - Pipeline(scaler + LR)            → LinearExplainer on scaled features,
                                          but raw values shown in plots for
                                          readability.
    """
    if isinstance(model, Pipeline):
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]

        bg = scaler.transform(
            X_train.sample(min(BG_SAMPLE_SIZE, len(X_train)), random_state=42)
        )
        bg_df = pd.DataFrame(bg, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        explainer = shap.LinearExplainer(clf, bg_df)
        shap_values = explainer(X_test_scaled)

        # Show *raw* feature values in plots — much more readable than
        # standardised z-scores. SHAP values are still computed correctly
        # in scaled space; we just override the displayed data.
        shap_values.data = X_test.values
        shap_values.feature_names = list(X_test.columns)
        return shap_values

    # Tree-based model — TreeSHAP is exact and runs in milliseconds.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Some XGBoost versions return (n, p, 2) for binary classification;
    # slice the positive-class column so downstream plots see (n, p).
    if shap_values.values.ndim == 3:
        shap_values = shap_values[..., 1]
    return shap_values


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

def _save_current_figure(save_path: Path, w: float = 9.0, h: float = 7.0) -> None:
    """shap.plots.* create their own figure; we just resize + save it."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.set_size_inches(w, h)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_global_bar(shap_values: shap.Explanation, save_path: Path) -> None:
    shap.plots.bar(shap_values, max_display=TOP_N_FEATURES, show=False)
    plt.title(f"Top {TOP_N_FEATURES} features — mean(|SHAP|)")
    _save_current_figure(save_path, w=8, h=7)


def plot_beeswarm(shap_values: shap.Explanation, save_path: Path) -> None:
    shap.plots.beeswarm(shap_values, max_display=TOP_N_FEATURES, show=False)
    plt.title("SHAP summary (beeswarm)")
    _save_current_figure(save_path, w=9, h=7)


def plot_waterfall(shap_values: shap.Explanation, idx: int, save_path: Path,
                   subtitle: str = "") -> None:
    shap.plots.waterfall(shap_values[idx], max_display=TOP_N_FEATURES, show=False)
    plt.title(f"Per-customer SHAP — test row {idx}\n{subtitle}", fontsize=11)
    _save_current_figure(save_path, w=10, h=7)


# --------------------------------------------------------------------------- #
# Pick an interesting customer for the waterfall
# --------------------------------------------------------------------------- #

def pick_demo_customer(model, X_test: pd.DataFrame, y_test: pd.Series,
                       target: float = WATERFALL_TARGET_PROBA) -> int:
    """Among actual churners, pick the test row whose predicted probability is
    closest to `target`. Gives a clean true-positive example with meaningful
    contributions in both directions."""
    proba = model.predict_proba(X_test)[:, 1]
    churner_mask = y_test.to_numpy() == 1
    distances = np.where(churner_mask, np.abs(proba - target), np.inf)
    return int(np.argmin(distances))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model, metadata, X_train, X_test, y_test = load_artefacts()
    log.info(
        "Winner: %s (val AUC %.4f, test AUC %.4f)",
        metadata["best_model"],
        metadata.get("best_val_auc", float("nan")),
        metadata.get("test_metrics", {}).get("auc", float("nan")),
    )
    log.info("X_test shape: %s", X_test.shape)

    log.info("Computing SHAP values …")
    shap_values = compute_shap_values(model, X_train, X_test)
    log.info("SHAP values shape: %s", shap_values.values.shape)

    # ----- Pick a demo customer for the waterfall ----- #
    idx = pick_demo_customer(model, X_test, y_test)
    proba = float(model.predict_proba(X_test.iloc[[idx]])[0, 1])
    actual = int(y_test.iloc[idx])
    log.info(
        "Waterfall customer: row %d  predicted churn=%.3f  actual=%d",
        idx, proba, actual,
    )

    # ----- Top-5 global feature importance to console ----- #
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top5 = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=False).head(5)
    log.info("Top-5 features by mean(|SHAP|):")
    for name, val in top5.items():
        log.info("  %-30s  %.4f", name, val)

    # ----- Save the three required plots ----- #
    plot_global_bar(shap_values, FIG_DIR / "shap_bar_top15.png")
    plot_beeswarm(shap_values, FIG_DIR / "shap_beeswarm.png")
    subtitle = f"predicted = {proba:.2f}  |  actual = {'Churned' if actual else 'Stayed'}"
    plot_waterfall(shap_values, idx, FIG_DIR / f"shap_waterfall_row{idx}.png", subtitle=subtitle)
    log.info("Saved 3 SHAP plots → %s", FIG_DIR)

    # ----- Cache SHAP values for the dashboard (avoids recomputation) ----- #
    cache_path = MODELS_DIR / "shap_cache.pkl"
    joblib.dump(
        {
            "shap_values": shap_values,
            "X_test": X_test,
            "y_test": y_test,
            "model_name": metadata["best_model"],
            "demo_customer_idx": idx,
        },
        cache_path,
    )
    log.info("Saved SHAP cache → %s", cache_path)

    log.info("Step 3 complete.")


if __name__ == "__main__":
    main()
