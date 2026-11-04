"""Streamlit dashboard for the student academic-outcome classifier.

Upload a test CSV (input features + the ``Target`` column), choose one of the
five saved classifiers or compare them all, and the app loads the matching
pre-trained pipeline and reports accuracy, AUC, precision, recall, F1 and MCC
alongside a confusion matrix and a per-class classification report.

No model is ever retrained here - the saved ``model/*.joblib`` pipelines are
loaded and used for inference only.

Results are rendered as plain HTML tables and inline (base64) images through the
core markdown component, instead of Streamlit's lazily-loaded Arrow/Vega
widgets. That keeps the output visible even on locked-down corporate networks
that block dynamically imported JavaScript chunks.

    streamlit run app.py
"""
from __future__ import annotations

import base64
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from src import config  # noqa: E402
from src.evaluation import load_saved_models, score_model  # noqa: E402
from src.features import FEATURE_COLUMNS  # noqa: E402

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Outcome Classifier", page_icon="🎓", layout="wide")

COMPARE_ALL = "Compare all models"
METRIC_HELP = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


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


def html_table(frame: pd.DataFrame, highlight_max: bool = False,
               value_format: str | None = None) -> str:
    """Render a DataFrame as an inline-styled HTML table (no JS widget).

    Column headers and the row-index labels are always bold. When
    ``highlight_max`` is set, the largest value in each numeric column is shown
    with a green, bold cell.
    """
    max_mask = None
    if highlight_max:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        max_mask = numeric.eq(numeric.max())

    def cell(value: object) -> str:
        if value_format is not None and isinstance(value, float):
            return value_format.format(value)
        return "" if pd.isna(value) else str(value)

    header = "".join(
        f'<th style="padding:6px 12px;background:#eef1f5;font-weight:700;'
        f'border:1px solid #dfe3e8;text-align:center">{col}</th>'
        for col in frame.columns
    )
    body = ""
    for idx, row in frame.iterrows():
        cells = ""
        for col in frame.columns:
            style = "padding:6px 12px;border:1px solid #eceff3;text-align:center;"
            if max_mask is not None and bool(max_mask.loc[idx, col]):
                style += "background:#b7e4c7;font-weight:700;"
            cells += f'<td style="{style}">{cell(row[col])}</td>'
        body += (f'<tr><th style="padding:6px 12px;border:1px solid #eceff3;'
                 f'background:#fafbfc;font-weight:700;text-align:left">{idx}</th>'
                 f'{cells}</tr>')

    table = ('<table style="border-collapse:collapse;width:100%;font-size:0.9rem">'
             f'<tr><th style="background:#eef1f5;border:1px solid #dfe3e8"></th>'
             f'{header}</tr>{body}</table>')
    return f'<div style="overflow-x:auto">{table}</div>'


def figure_to_html(fig, full_width: bool = False) -> str:
    """Embed a matplotlib figure inline as a base64 <img> (no media fetch)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    width = "width:100%;" if full_width else "max-width:100%;"
    return (f'<img src="data:image/png;base64,{encoded}" '
            f'style="{width}height:auto;"/>')


def metric_cards(scores: dict) -> str:
    """Render the six metrics as a row of HTML cards."""
    cards = "".join(
        '<div style="flex:1;min-width:95px;text-align:center;padding:10px;'
        'border:1px solid #e6e6e6;border-radius:10px;background:#fafafa">'
        f'<div style="font-size:0.78em;color:#555">{name}</div>'
        f'<div style="font-size:1.45em;font-weight:700">{scores[name]:.4f}</div></div>'
        for name in METRIC_HELP
    )
    return f'<div style="display:flex;gap:8px;flex-wrap:wrap">{cards}</div>'


def confusion_figure(model, features, target):
    matrix = confusion_matrix(target, model.predict(features), labels=config.CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=config.CLASS_ORDER, yticklabels=config.CLASS_ORDER, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def classification_frame(model, features, target) -> pd.DataFrame:
    report = classification_report(
        target, model.predict(features),
        labels=config.CLASS_ORDER, output_dict=True, zero_division=0,
    )
    return pd.DataFrame(report).T.round(3)


def comparison_bar_figure(table: pd.DataFrame):
    columns = ["Accuracy", "F1", "MCC"]
    fig, ax = plt.subplots(figsize=(16, 4.5))
    table[columns].plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(table.index, rotation=0)
    # Legend parked in the empty top-right area so it never covers the bars.
    ax.legend(loc="upper right", ncol=3, fontsize=11, framealpha=0.9)
    fig.tight_layout()
    return fig


def all_confusions_figure(models, features, target):
    """Confusion matrices for every model on one figure (for the compare view)."""
    count = len(models)
    cols = 3
    grid_rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(grid_rows, cols, figsize=(13, 4.0 * grid_rows))
    axes = axes.ravel()
    for ax, (_, (display, model)) in zip(axes, models.items()):
        matrix = confusion_matrix(target, model.predict(features), labels=config.CLASS_ORDER)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=config.CLASS_ORDER, yticklabels=config.CLASS_ORDER, ax=ax)
        ax.set_title(display, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes[count:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🎓 Student Academic Outcome Classifier")
st.caption(
    "Score pre-trained models on a test set - Dropout vs Enrolled vs Graduate. "
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
    balance = target.value_counts().reindex(config.CLASS_ORDER).fillna(0).astype(int)
    balance.loc["Total"] = int(balance.sum())
    st.write(f"**Shape:** {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    st.caption("First 10 rows")
    st.markdown(html_table(frame.head(10).reset_index(drop=True)),
                unsafe_allow_html=True)
    st.caption("Class balance")
    st.markdown(html_table(balance.rename("count").to_frame()), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if choice == COMPARE_ALL:
    st.subheader("Model comparison on this test set")
    rows = {display: score_model(model, features, target)
            for _, (display, model) in models.items()}
    table = pd.DataFrame(rows).T[METRIC_HELP].round(4)
    st.markdown(html_table(table, highlight_max=True, value_format="{:.4f}"),
                unsafe_allow_html=True)
    st.caption("Higher is better for every metric.")
    st.markdown(figure_to_html(comparison_bar_figure(table), full_width=True),
                unsafe_allow_html=True)

    st.markdown("**Confusion matrix — all models**")
    st.markdown(figure_to_html(all_confusions_figure(models, features, target)),
                unsafe_allow_html=True)

    st.markdown("**Classification report — per model**")
    for _, (display, model) in models.items():
        st.caption(display)
        st.markdown(html_table(classification_frame(model, features, target),
                               value_format="{:.3f}"), unsafe_allow_html=True)
else:
    key = next(k for k, (display, _) in models.items() if display == choice)
    _, model = models[key]
    scores = score_model(model, features, target)

    st.subheader(f"{choice} - evaluation metrics")
    st.markdown(metric_cards(scores), unsafe_allow_html=True)
    st.write("")

    panel_left, panel_right = st.columns(2)
    with panel_left:
        st.markdown("**Confusion matrix**")
        st.markdown(figure_to_html(confusion_figure(model, features, target)),
                    unsafe_allow_html=True)
    with panel_right:
        st.markdown("**Classification report**")
        st.markdown(html_table(classification_frame(model, features, target),
                               value_format="{:.3f}"), unsafe_allow_html=True)

st.divider()
st.caption("UCI dataset #697 · five scikit-learn pipelines · Machine Learning Assignment 2")
