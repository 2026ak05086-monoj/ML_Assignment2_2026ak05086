"""Streamlit dashboard for the student academic-outcome classifier.

Upload a test CSV (input features + the ``Target`` column), choose one of the
five saved classifiers or compare them all, and the app loads the matching
pre-trained pipeline and reports accuracy, AUC, precision, recall, F1 and MCC
alongside a confusion matrix and per-class classification report.

No model is ever retrained here - the saved ``model/*.joblib`` pipelines are
loaded and only used for inference.

    streamlit run app.py
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.evaluation import load_saved_models, score_model
from src.features import FEATURE_COLUMNS

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Outcome Classifier", page_icon="🎓", layout="wide")

COMPARE_ALL = "📊 Compare all models"
METRIC_HELP = {
    "Accuracy": "Share of students classified correctly.",
    "AUC": "Weighted one-vs-rest ROC-AUC.",
    "Precision": "Weighted precision across the three outcomes.",
    "Recall": "Weighted recall across the three outcomes.",
    "F1": "Weighted harmonic mean of precision and recall.",
    "MCC": "Matthews correlation coefficient (ranges -1 to 1).",
}


@st.cache_resource(show_spinner="Loading trained models…")
def get_models():
    return load_saved_models()


def split_features_target(frame: pd.DataFrame):
    """Return (X, y, missing_columns) after validating the uploaded frame."""
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if config.TARGET_COLUMN not in frame.columns:
        missing = missing + [config.TARGET_COLUMN]
    if missing:
        return None, None, missing
    return frame[FEATURE_COLUMNS].copy(), frame[config.TARGET_COLUMN].copy(), []


def confusion_figure(model, features, target):
    matrix = confusion_matrix(target, model.predict(features), labels=config.CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=config.CLASS_ORDER, yticklabels=config.CLASS_ORDER, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def classification_table(model, features, target) -> pd.DataFrame:
    report = classification_report(
        target, model.predict(features),
        labels=config.CLASS_ORDER, output_dict=True, zero_division=0,
    )
    return pd.DataFrame(report).T.round(3)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🎓 Student Academic Outcome Classifier")
st.caption(
    "Score pre-trained models on a test set — Dropout vs Enrolled vs Graduate. "
    "Models are loaded from disk and used for inference only (no retraining)."
)

models = get_models()
if not models:
    st.error("No trained models were found in `model/`. "
             "Run `python -m src.tuning <model>` first.")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload a CSV (features + Target)", type=["csv"])
    use_sample = st.checkbox("Use the bundled sample (test_data.csv)", value=True)

    st.header("2 · Model")
    display_names = [display for _, (display, _) in models.items()]
    choice = st.selectbox("Classifier", [COMPARE_ALL, *display_names])

    st.divider()
    st.caption("Precision / Recall / F1 / AUC use weighted averaging; AUC is one-vs-rest.")

# --------------------------------------------------------------------------- #
# Resolve the data source (an upload always wins over the sample)
# --------------------------------------------------------------------------- #
frame, source_label = None, None
if uploaded is not None:
    frame = pd.read_csv(uploaded)
    source_label = f"uploaded file · {uploaded.name}"
elif use_sample and config.TEST_DATA_FILE.exists():
    frame = pd.read_csv(config.TEST_DATA_FILE)
    source_label = "bundled sample · test_data.csv"

if frame is None:
    st.info("⬅️ Upload a test CSV, or tick **Use the bundled sample** to get started.")
    with st.expander("What should the CSV contain?"):
        st.write(f"- All **{len(FEATURE_COLUMNS)}** input feature columns.")
        st.write(f"- A **`{config.TARGET_COLUMN}`** column with values: "
                 f"{', '.join(config.CLASS_ORDER)}.")
    st.stop()

features, target, missing = split_features_target(frame)
if missing:
    shown = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
    st.error(f"The uploaded CSV is missing required column(s): {shown}")
    st.stop()

st.success(f"Loaded **{len(frame):,}** rows from {source_label}.")
with st.expander("Preview data & class balance"):
    left, right = st.columns([3, 1])
    left.dataframe(frame.head(10), width="stretch")
    balance = target.value_counts().reindex(config.CLASS_ORDER).rename("count")
    right.write("**Class balance**")
    right.dataframe(balance)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if choice == COMPARE_ALL:
    st.subheader("Model comparison on this test set")
    rows = {display: score_model(model, features, target)
            for _, (display, model) in models.items()}
    table = pd.DataFrame(rows).T[list(METRIC_HELP)].round(4)
    st.dataframe(table.style.highlight_max(axis=0, color="#b7e4c7"),
                 width="stretch")
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
    st.caption("Highlighted cells mark the best score for each metric.")
else:
    key = next(k for k, (display, _) in models.items() if display == choice)
    _, model = models[key]
    scores = score_model(model, features, target)

    st.subheader(f"{choice} — evaluation metrics")
    for metric_col, (name, value) in zip(st.columns(len(scores)), scores.items()):
        metric_col.metric(name, f"{value:.4f}", help=METRIC_HELP.get(name))

    panel_left, panel_right = st.columns(2)
    with panel_left:
        st.markdown("**Confusion matrix**")
        st.pyplot(confusion_figure(model, features, target))
    with panel_right:
        st.markdown("**Classification report**")
        st.dataframe(classification_table(model, features, target),
                     width="stretch")

st.divider()
st.caption("UCI dataset #697 · five scikit-learn pipelines · Machine Learning Assignment 2")
