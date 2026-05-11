"""
Customer Churn Prediction Dashboard.

A 4-page Streamlit app that surfaces the model trained in Steps 1-3:

    Page 1  Overview                — descriptive KPIs from the raw data
    Page 2  Model Performance      — ROC / PR / confusion matrix + SHAP global importance
    Page 3  Predict Single Customer — interactive form → probability + waterfall
    Page 4  Batch Prediction       — upload CSV → predictions + top-10 risk + download

Run from the project root:

    streamlit run app/streamlit_app.py

Prerequisites (run these once, in order):

    python -m src.data_pipeline   # writes data/processed/*.csv
    python -m src.train           # writes models/best_model.pkl, models/best_metadata.json
    python -m src.explain         # writes models/shap_cache.pkl, reports/figures/*.png
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------- #
# Path bootstrap — make `src.*` importable when Streamlit launches this file
# from anywhere. Streamlit's CWD is the project root in practice, but we still
# inject the project root onto sys.path so `from src...` works deterministically.
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import (  # noqa: E402  (import after sys.path edit)
    TARGET_COL,
    clean,
    encode,
    engineer_features,
)
from src.explain import compute_shap_values  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths + constants
# --------------------------------------------------------------------------- #

RAW_CSV = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
META_PATH = MODELS_DIR / "best_metadata.json"
SHAP_CACHE_PATH = MODELS_DIR / "shap_cache.pkl"
FEATURE_COLS_PATH = PROCESSED_DIR / "feature_columns.json"

RISK_THRESHOLDS = (0.33, 0.66)        # (low_max, medium_max) — anything above 0.66 is High
DEFAULT_DECISION_THRESHOLD = 0.5      # used for confusion matrix etc.
TOP_N_FEATURES = 15

# --------------------------------------------------------------------------- #
# Streamlit page config
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading trained model…")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    with META_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_columns() -> list[str]:
    with FEATURE_COLS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading raw Telco dataset…")
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    # Normalise the dollar column once — the raw file ships TotalCharges as
    # object (blank strings for tenure==0 customers).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


@st.cache_data(show_spinner="Loading test split…")
def load_test_split() -> tuple[pd.DataFrame, pd.Series]:
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_test, y_test


@st.cache_data(show_spinner=False)
def load_train_features() -> pd.DataFrame:
    """Background sample for SHAP — only the encoded feature matrix is needed."""
    return pd.read_csv(PROCESSED_DIR / "X_train.csv")


@st.cache_resource(show_spinner="Loading SHAP cache…")
def load_shap_cache() -> dict:
    return joblib.load(SHAP_CACHE_PATH)


@st.cache_resource(show_spinner="Building SHAP explainer…")
def get_runtime_explainer(_model, X_train_bg: pd.DataFrame):
    """Build an explainer once per session for the live single-customer flow.

    The leading underscore on `_model` tells Streamlit not to hash it — joblib
    estimators are not always hashable, but they're already pinned by load_model().
    """
    bg = X_train_bg.sample(min(200, len(X_train_bg)), random_state=42)
    # Reuse the project's helper so the runtime explainer matches train-time logic.
    # compute_shap_values handles both Pipeline(LR) and tree models.
    # We call it with a small background to keep TreeSHAP fast even for LinearExplainer.
    return _SharedExplainer(_model, bg)


class _SharedExplainer:
    """Thin wrapper that builds the right SHAP explainer once and exposes
    `.explain(X)` returning a `shap.Explanation`. Mirrors the dispatch logic
    in `src.explain.compute_shap_values` so plots match training-time output.
    """

    def __init__(self, model, X_train_bg: pd.DataFrame):
        from sklearn.pipeline import Pipeline

        self.model = model
        self.is_pipeline = isinstance(model, Pipeline)

        if self.is_pipeline:
            self.scaler = model.named_steps["scaler"]
            clf = model.named_steps["clf"]
            bg_scaled = pd.DataFrame(
                self.scaler.transform(X_train_bg), columns=X_train_bg.columns
            )
            self._explainer = shap.LinearExplainer(clf, bg_scaled)
            self._columns = list(X_train_bg.columns)
        else:
            self._explainer = shap.TreeExplainer(model)
            self._columns = list(X_train_bg.columns)

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        if self.is_pipeline:
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            exp = self._explainer(X_scaled)
            # Show raw values in plots, not z-scores.
            exp.data = X.values
            exp.feature_names = self._columns
            return exp

        exp = self._explainer(X)
        if exp.values.ndim == 3:        # XGBoost (n, p, 2) binary case
            exp = exp[..., 1]
        return exp


# --------------------------------------------------------------------------- #
# Domain helpers
# --------------------------------------------------------------------------- #

def encode_customer_record(record: dict) -> pd.DataFrame:
    """Run a single raw input dict through the training pipeline so the result
    is aligned to `feature_columns.json`. Reuses src.data_pipeline.* so dashboard
    encoding can never silently drift from training.
    """
    feature_cols = load_feature_columns()
    df = pd.DataFrame([record])
    # `clean` would coerce Churn — add a stub so it survives, then drop it.
    df[TARGET_COL] = "No"
    df = clean(df)
    df = engineer_features(df)
    df = encode(df)
    df = df.drop(columns=[TARGET_COL], errors="ignore")

    # Add any missing dummy columns (a single record can't produce every level)
    # and drop unexpected ones, then reorder.
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    return df


def encode_batch(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Same idea as encode_customer_record but for an uploaded CSV."""
    feature_cols = load_feature_columns()
    df = raw_df.copy()
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = "No"        # stub so `clean` succeeds
    df = clean(df)
    df = engineer_features(df)
    df = encode(df)
    df = df.drop(columns=[TARGET_COL], errors="ignore")

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


def risk_label(p: float) -> tuple[str, str]:
    low_max, med_max = RISK_THRESHOLDS
    if p < low_max:
        return "Low", "#2ecc71"
    if p < med_max:
        return "Medium", "#f39c12"
    return "High", "#e74c3c"


# --------------------------------------------------------------------------- #
# Plot helpers (Plotly)
# --------------------------------------------------------------------------- #

def plotly_roc(y_true, y_proba) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
                             line=dict(color="#2e86de", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color="gray", dash="dash")))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        legend=dict(x=0.55, y=0.05),
    )
    return fig


def plotly_pr(y_true, y_proba) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(np.mean(y_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                             name=f"AP = {ap:.3f}", line=dict(color="#8e44ad", width=3)))
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                  annotation_text=f"base rate = {baseline:.3f}")
    fig.update_layout(
        title="Precision–Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=420,
        legend=dict(x=0.05, y=0.05),
    )
    return fig


def plotly_confusion(y_true, y_pred) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Stayed", "Churned"]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels, y=labels,
        text=cm, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix (threshold = {DEFAULT_DECISION_THRESHOLD})",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=420,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plotly_gauge(probability: float) -> go.Figure:
    label, color = risk_label(probability)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number=dict(suffix=" %", font=dict(size=42)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=color, thickness=0.35),
            steps=[
                {"range": [0,  RISK_THRESHOLDS[0] * 100], "color": "#eafaf1"},
                {"range": [RISK_THRESHOLDS[0] * 100, RISK_THRESHOLDS[1] * 100], "color": "#fef5e7"},
                {"range": [RISK_THRESHOLDS[1] * 100, 100], "color": "#fdedec"},
            ],
            threshold=dict(line=dict(color=color, width=4),
                           thickness=0.85, value=probability * 100),
        ),
        title=dict(text=f"Churn risk: <b>{label}</b>", font=dict(size=18)),
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plotly_global_importance(shap_values: shap.Explanation, top_n: int = TOP_N_FEATURES) -> go.Figure:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = (
        pd.Series(mean_abs, index=shap_values.feature_names)
        .sort_values(ascending=True)
        .tail(top_n)
    )
    fig = go.Figure(go.Bar(
        x=importance.values, y=importance.index, orientation="h",
        marker=dict(color="#2e86de"),
    ))
    fig.update_layout(
        title=f"Top {top_n} features (mean |SHAP|)",
        xaxis_title="mean(|SHAP value|)",
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def render_shap_waterfall_native(exp: shap.Explanation) -> None:
    """SHAP's matplotlib waterfall — rendered via st.pyplot."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(exp, max_display=TOP_N_FEATURES, show=False)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Guard — show a friendly setup message if artefacts are missing
# --------------------------------------------------------------------------- #

def require_artefacts() -> bool:
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH.relative_to(ROOT)))
    if not META_PATH.exists():
        missing.append(str(META_PATH.relative_to(ROOT)))
    if not FEATURE_COLS_PATH.exists():
        missing.append(str(FEATURE_COLS_PATH.relative_to(ROOT)))

    if not missing:
        return True

    st.error("Required model artefacts are missing.")
    st.markdown(
        "Run the training pipeline first, in order, from the project root:\n\n"
        "```bash\n"
        "python -m src.data_pipeline\n"
        "python -m src.train\n"
        "python -m src.explain\n"
        "```\n\n"
        "**Missing files:**\n"
        + "\n".join(f"- `{m}`" for m in missing)
    )
    return False


# --------------------------------------------------------------------------- #
# Page 1 — Overview
# --------------------------------------------------------------------------- #

def page_overview() -> None:
    st.title(":bar_chart: Customer Churn — Overview")
    st.caption("Descriptive view of the IBM Telco Customer Churn dataset.")

    if not RAW_CSV.exists():
        st.warning(
            f"Raw dataset not found at `{RAW_CSV.relative_to(ROOT)}`. "
            "See README for the Kaggle download."
        )
        return

    df = load_raw()
    churned = df[df[TARGET_COL].str.lower() == "yes"]
    total_customers = len(df)
    churn_rate = len(churned) / total_customers
    revenue_at_risk = churned["MonthlyCharges"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total customers", f"{total_customers:,}")
    c2.metric("Churn rate", f"{churn_rate*100:.1f} %")
    c3.metric("Revenue at risk (monthly)", f"${revenue_at_risk:,.0f}")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Churn distribution")
        counts = df[TARGET_COL].value_counts().rename({"Yes": "Churned", "No": "Stayed"})
        pie = px.pie(
            names=counts.index, values=counts.values, hole=0.45,
            color=counts.index,
            color_discrete_map={"Churned": "#e74c3c", "Stayed": "#2ecc71"},
        )
        pie.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(pie, use_container_width=True)

    with right:
        st.subheader("Churn by contract type")
        grouped = (
            df.groupby(["Contract", TARGET_COL]).size()
            .unstack(fill_value=0)
            .rename(columns={"Yes": "Churned", "No": "Stayed"})
            .reset_index()
        )
        bar = go.Figure()
        bar.add_trace(go.Bar(x=grouped["Contract"], y=grouped["Stayed"],
                             name="Stayed", marker_color="#2ecc71"))
        bar.add_trace(go.Bar(x=grouped["Contract"], y=grouped["Churned"],
                             name="Churned", marker_color="#e74c3c"))
        bar.update_layout(
            barmode="stack", height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Contract type", yaxis_title="Customers",
        )
        st.plotly_chart(bar, use_container_width=True)

    with st.expander("Why month-to-month customers dominate the at-risk pool"):
        st.write(
            "Month-to-month contracts have an order-of-magnitude higher churn rate "
            "than 1- or 2-year contracts. Page 2 confirms this empirically: contract "
            "type is the single biggest SHAP feature for the trained model."
        )


# --------------------------------------------------------------------------- #
# Page 2 — Model Performance
# --------------------------------------------------------------------------- #

def page_performance() -> None:
    st.title(":dart: Model Performance")
    if not require_artefacts():
        return

    meta = load_metadata()
    model = load_model()
    X_test, y_test = load_test_split()

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DEFAULT_DECISION_THRESHOLD).astype(int)
    test = meta.get("test_metrics", {})

    st.subheader(f"Winner: `{meta['best_model']}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test AUC",       f"{test.get('auc', 0):.4f}")
    c2.metric("Avg precision",  f"{test.get('avg_precision', 0):.4f}")
    c3.metric("F1",             f"{test.get('f1', 0):.4f}")
    c4.metric("Recall",         f"{test.get('recall', 0):.4f}")

    if "all_test_metrics" in meta:
        st.caption("All three models on the held-out test set:")
        comp = (
            pd.DataFrame(meta["all_test_metrics"]).T
            .reset_index().rename(columns={"index": "model"})
            .sort_values("auc", ascending=False)
        )
        st.dataframe(
            comp.style.format({c: "{:.4f}" for c in comp.columns if c != "model"}),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plotly_roc(y_test, y_proba), use_container_width=True)
    with col2:
        st.plotly_chart(plotly_pr(y_test, y_proba), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plotly_confusion(y_test, y_pred), use_container_width=True)
    with col4:
        st.subheader("Global feature importance (SHAP)")
        if SHAP_CACHE_PATH.exists():
            cache = load_shap_cache()
            st.plotly_chart(
                plotly_global_importance(cache["shap_values"]),
                use_container_width=True,
            )
        else:
            st.info(
                "SHAP cache not found. Run `python -m src.explain` to populate "
                "`models/shap_cache.pkl`."
            )


# --------------------------------------------------------------------------- #
# Page 3 — Predict Single Customer
# --------------------------------------------------------------------------- #

# Raw schema reference — fed to the input form. Keep aligned with the Telco CSV.
_YES_NO = ["No", "Yes"]
_YES_NO_SERVICE = ["No", "Yes", "No internet service"]
_YES_NO_PHONE = ["No", "Yes", "No phone service"]
_CONTRACT = ["Month-to-month", "One year", "Two year"]
_PAYMENT = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]
_INTERNET = ["DSL", "Fiber optic", "No"]


def _customer_input_form() -> dict | None:
    """Render the form and return a raw-schema record dict on submit."""
    with st.form("single_customer", border=True):
        st.markdown("**Customer profile**")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Partner", _YES_NO)
            dependents = st.selectbox("Dependents", _YES_NO)
        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5)
            total = st.number_input("Total charges ($)", min_value=0.0, value=float(tenure * monthly), step=10.0)
        with c3:
            contract = st.selectbox("Contract", _CONTRACT)
            paperless = st.selectbox("Paperless billing", _YES_NO)
            payment = st.selectbox("Payment method", _PAYMENT)

        st.markdown("**Services**")
        s1, s2, s3 = st.columns(3)
        with s1:
            phone = st.selectbox("Phone service", _YES_NO)
            multilines = st.selectbox("Multiple lines", _YES_NO_PHONE)
            internet = st.selectbox("Internet service", _INTERNET)
        with s2:
            online_sec = st.selectbox("Online security", _YES_NO_SERVICE)
            online_bak = st.selectbox("Online backup", _YES_NO_SERVICE)
            device_pro = st.selectbox("Device protection", _YES_NO_SERVICE)
        with s3:
            tech_sup = st.selectbox("Tech support", _YES_NO_SERVICE)
            stream_tv = st.selectbox("Streaming TV", _YES_NO_SERVICE)
            stream_mv = st.selectbox("Streaming movies", _YES_NO_SERVICE)

        submitted = st.form_submit_button("Predict churn", type="primary",
                                          use_container_width=True)

    if not submitted:
        return None

    return {
        "gender": gender, "SeniorCitizen": int(senior),
        "Partner": partner, "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone, "MultipleLines": multilines,
        "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_bak,
        "DeviceProtection": device_pro, "TechSupport": tech_sup,
        "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly), "TotalCharges": float(total),
    }


def page_single_predict() -> None:
    st.title(":mag: Predict — Single Customer")
    if not require_artefacts():
        return

    record = _customer_input_form()
    if record is None:
        st.info("Fill in the form and click **Predict churn** to see a probability and SHAP explanation.")
        return

    model = load_model()
    X_row = encode_customer_record(record)
    probability = float(model.predict_proba(X_row)[0, 1])
    label, color = risk_label(probability)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(plotly_gauge(probability), use_container_width=True)
        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background:{color}22;"
            f"border-left:6px solid {color};'>"
            f"<b style='color:{color};font-size:18px;'>{label} risk</b> &nbsp;—&nbsp;"
            f"predicted churn probability <b>{probability:.1%}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Why the model predicts this")
        try:
            explainer = get_runtime_explainer(model, load_train_features())
            exp = explainer.explain(X_row)
            render_shap_waterfall_native(exp[0])
        except Exception as e:        # SHAP can fail on odd model/feature combos
            st.warning(f"Could not render SHAP explanation: {e}")


# --------------------------------------------------------------------------- #
# Page 4 — Batch Prediction
# --------------------------------------------------------------------------- #

_REQUIRED_RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def page_batch_predict() -> None:
    st.title(":inbox_tray: Batch Prediction")
    if not require_artefacts():
        return

    st.markdown(
        "Upload a CSV with the **raw Telco schema** (same columns as the Kaggle "
        "dataset minus `Churn`). The app will encode, score, and rank every row."
    )
    with st.expander("Required columns"):
        st.code(", ".join(_REQUIRED_RAW_COLS))

    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    if uploaded is None:
        st.info("Tip: drop in `data/raw/telco_churn.csv` to try it on the full dataset.")
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return

    missing_cols = [c for c in _REQUIRED_RAW_COLS if c not in raw.columns]
    if missing_cols:
        st.error("CSV is missing required columns: " + ", ".join(missing_cols))
        return

    model = load_model()
    try:
        X = encode_batch(raw)
    except Exception as e:
        st.error(f"Encoding failed: {e}")
        return

    proba = model.predict_proba(X)[:, 1]
    out = raw.copy()
    out["churn_probability"] = proba
    out["risk"] = pd.Series(proba).map(lambda p: risk_label(p)[0]).values
    out["prediction"] = (proba >= DEFAULT_DECISION_THRESHOLD).astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", f"{len(out):,}")
    c2.metric("Predicted churners", f"{int(out['prediction'].sum()):,}")
    c3.metric("Avg churn probability", f"{proba.mean():.1%}")

    st.divider()

    st.subheader("Top 10 highest-risk customers")
    top_cols = ["churn_probability", "risk", "tenure", "Contract",
                "MonthlyCharges", "PaymentMethod"]
    if "customerID" in out.columns:
        top_cols = ["customerID"] + top_cols
    top10 = out.sort_values("churn_probability", ascending=False).head(10)[top_cols]
    st.dataframe(
        top10.style.format({"churn_probability": "{:.1%}", "MonthlyCharges": "${:.2f}"}),
        use_container_width=True, hide_index=True,
    )

    st.subheader("All predictions")
    st.dataframe(
        out.style.format({"churn_probability": "{:.1%}"}),
        use_container_width=True, hide_index=True, height=350,
    )

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button(
        "Download predictions as CSV",
        data=buf.getvalue(),
        file_name="churn_predictions.csv",
        mime="text/csv",
        type="primary",
    )


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.title("Churn Dashboard")
    st.caption("Trained on the IBM Telco Customer Churn dataset.")
    if META_PATH.exists():
        meta = load_metadata()
        st.markdown(
            f"**Model:** `{meta['best_model']}`  \n"
            f"**Test AUC:** `{meta.get('test_metrics', {}).get('auc', 0):.4f}`"
        )
    st.divider()
    st.caption("Steps 1-3 must be run before launching this app — see the README.")

nav = st.navigation([
    st.Page(page_overview,        title="Overview",            icon=":material/dashboard:"),
    st.Page(page_performance,     title="Model Performance",   icon=":material/analytics:"),
    st.Page(page_single_predict,  title="Predict Customer",    icon=":material/person_search:"),
    st.Page(page_batch_predict,   title="Batch Prediction",    icon=":material/upload_file:"),
])
nav.run()
