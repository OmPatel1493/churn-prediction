"""
Step 2 — Evaluation utilities: metrics + diagnostic plots.

Pure functions, no side effects beyond writing PNGs to disk. Imported by
src/train.py and src/explain.py and (later) the Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #

def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Headline classification metrics. All values cast to plain floats so the
    dict is JSON-serialisable."""
    return {
        "auc":           float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "f1":            float(f1_score(y_true, y_pred)),
        "precision":     float(precision_score(y_true, y_pred)),
        "recall":        float(recall_score(y_true, y_pred)),
        "accuracy":      float(accuracy_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_roc_curve(y_true, y_proba, save_path: Path, title: str = "ROC Curve") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true, y_proba, save_path: Path, title: str = "Precision-Recall Curve") -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    # Precision baseline equals the positive-class prevalence in y_true.
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(baseline, ls="--", color="k", lw=1, label=f"Base rate = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, save_path: Path, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"],
        ax=ax, cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# High-level helper
# --------------------------------------------------------------------------- #

def evaluate_model(model, X, y, name: str, fig_dir: Path, threshold: float = 0.5) -> dict[str, float]:
    """Score `model` on (X, y), save 3 diagnostic plots under `fig_dir`,
    return the metrics dict.

    `name` is used as a filename prefix, e.g. 'xgboost' → xgboost_roc.png.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = compute_metrics(y, y_pred, y_proba)

    plot_roc_curve(y, y_proba, fig_dir / f"{name}_roc.png", title=f"ROC — {name}")
    plot_pr_curve(y, y_proba, fig_dir / f"{name}_pr.png", title=f"Precision-Recall — {name}")
    plot_confusion_matrix(y, y_pred, fig_dir / f"{name}_cm.png", title=f"Confusion Matrix — {name}")

    return metrics
