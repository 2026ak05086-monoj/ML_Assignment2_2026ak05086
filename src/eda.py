"""Exploratory data analysis for the student-outcome dataset.

Produces a handful of figures under ``reports/figures/`` and a short markdown
summary (``reports/eda_summary.md``) that later feeds the README's dataset
description and per-model observations.

    python -m src.eda
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # keep things headless-friendly

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.dataset import load_raw  # noqa: E402

sns.set_theme(style="whitegrid")

# Columns that live on a genuine numeric scale (worth correlating / plotting).
CONTINUOUS_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Features we expect to separate the three outcomes well.
SPOTLIGHT_FEATURES = [
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
]


def _plot_target_distribution(frame: pd.DataFrame, out_path: Path) -> None:
    counts = frame[config.TARGET_COLUMN].value_counts().reindex(config.CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=list(counts.index),
        y=counts.values,
        hue=list(counts.index),
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for i, value in enumerate(counts.values):
        ax.text(i, value + 15, str(int(value)), ha="center", fontweight="bold")
    ax.set_title("Academic outcome distribution")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Number of students")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_correlation(frame: pd.DataFrame, out_path: Path) -> None:
    corr = frame[CONTINUOUS_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation between continuous features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_spotlight_features(frame: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feature in zip(axes.ravel(), SPOTLIGHT_FEATURES):
        sns.boxplot(
            data=frame,
            x=config.TARGET_COLUMN,
            y=feature,
            order=config.CLASS_ORDER,
            hue=config.TARGET_COLUMN,
            palette="viridis",
            legend=False,
            ax=ax,
        )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Key features by academic outcome", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_summary(frame: pd.DataFrame, md_path: Path) -> None:
    counts = frame[config.TARGET_COLUMN].value_counts().reindex(config.CLASS_ORDER)
    total = int(counts.sum())
    dtype_counts = frame.dtypes.astype(str).value_counts().to_dict()

    lines = [
        "# Exploratory Data Analysis",
        "",
        f"- **Instances:** {frame.shape[0]}",
        f"- **Columns:** {frame.shape[1]} ({frame.shape[1] - 1} features + target)",
        f"- **Missing cells:** {int(frame.isna().sum().sum())}",
        f"- **Duplicate rows:** {int(frame.duplicated().sum())}",
        f"- **Column dtypes:** {dtype_counts}",
        "",
        "## Target balance",
        "",
    ]
    for cls in config.CLASS_ORDER:
        c = int(counts[cls])
        lines.append(f"- {cls}: {c} ({c / total:.1%})")
    lines += [
        "",
        "## Notes",
        "",
        "- Three-class, moderately imbalanced problem (Enrolled is the minority).",
        "- No missing values or duplicate rows, so cleaning effort is minimal.",
        "- Most integer columns are coded categoricals (marital status, course, "
        "parents' qualifications/occupations, binary flags); the grade, "
        "economic-indicator and age columns are genuinely continuous.",
        "- Approved curricular units (1st & 2nd semester) separate graduates from "
        "dropouts most clearly, so they should be strong predictors.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    frame = load_raw()
    config.ensure_directories()

    _plot_target_distribution(frame, config.FIGURE_DIR / "target_distribution.png")
    _plot_correlation(frame, config.FIGURE_DIR / "feature_correlation.png")
    _plot_spotlight_features(frame, config.FIGURE_DIR / "key_features_by_class.png")

    summary_path = config.REPORTS_DIR / "eda_summary.md"
    _write_summary(frame, summary_path)

    print("EDA complete.")
    print(f"Figures -> {config.FIGURE_DIR}")
    print(f"Summary -> {summary_path}")
    print("\nTarget balance:")
    print(frame[config.TARGET_COLUMN].value_counts().reindex(config.CLASS_ORDER))


if __name__ == "__main__":
    run()
