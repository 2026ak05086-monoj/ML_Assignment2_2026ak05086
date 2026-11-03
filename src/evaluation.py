"""Evaluate the five saved models on the held-out test split.

For every persisted model this computes the six metrics the assignment asks for
- accuracy, ROC-AUC, precision, recall, F1 and the Matthews correlation
coefficient - writes the comparison table to ``reports/model_scores.csv`` and a
grid of confusion matrices to ``reports/figures/confusion_matrices.png``.

Multiclass precision/recall/F1 and AUC use *weighted* averaging so that each
single number accounts for the class imbalance.

    python -m src.evaluation
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize  # noqa: E402

from src import config  # noqa: E402
from src.features import make_split  # noqa: E402

# Keep the SVC(probability=True) deprecation notice out of the report output.
warnings.filterwarnings("ignore", category=FutureWarning)

AVERAGE = "weighted"
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def load_saved_models() -> dict[str, tuple[str, object]]:
    """Return ``{key: (display_name, pipeline)}`` for every saved model."""
    loaded = {}
    for key, (display, filename) in config.MODEL_REGISTRY.items():
        path = config.MODEL_DIR / filename
        if path.exists():
            loaded[key] = (display, joblib.load(path))
    return loaded


def _auc_score(model, features, target) -> float:
    """Weighted one-vs-rest ROC-AUC with proba columns aligned to CLASS_ORDER."""
    proba = model.predict_proba(features)
    column_order = [list(model.classes_).index(c) for c in config.CLASS_ORDER]
    proba = proba[:, column_order]
    return roc_auc_score(
        target,
        proba,
        multi_class="ovr",
        average=AVERAGE,
        labels=config.CLASS_ORDER,
    )


def score_model(model, features, target) -> dict[str, float]:
    """Compute the six assignment metrics for one model on given data."""
    predicted = model.predict(features)
    return {
        "Accuracy": accuracy_score(target, predicted),
        "AUC": _auc_score(model, features, target),
        "Precision": precision_score(target, predicted, average=AVERAGE, zero_division=0),
        "Recall": recall_score(target, predicted, average=AVERAGE, zero_division=0),
        "F1": f1_score(target, predicted, average=AVERAGE, zero_division=0),
        "MCC": matthews_corrcoef(target, predicted),
    }


def _plot_confusions(models, features, target, out_path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    flat_axes = axes.ravel()
    for ax, (_, (display, model)) in zip(flat_axes, models.items()):
        matrix = confusion_matrix(target, model.predict(features), labels=config.CLASS_ORDER)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=config.CLASS_ORDER,
            yticklabels=config.CLASS_ORDER,
            ax=ax,
        )
        ax.set_title(display)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in flat_axes[len(models):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on the held-out test set", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate() -> pd.DataFrame:
    config.ensure_directories()
    _, x_test, _, y_test = make_split()
    models = load_saved_models()
    if not models:
        raise RuntimeError("No saved models found in model/. Train them first.")

    scores = {display: score_model(model, x_test, y_test)
              for _, (display, model) in models.items()}
    table = pd.DataFrame(scores).T[METRIC_ORDER].round(4)

    table.to_csv(config.REPORTS_DIR / "model_scores.csv")
    _plot_confusions(models, x_test, y_test,
                     config.FIGURE_DIR / "confusion_matrices.png")

    print(table.to_string())
    print(f"\nSaved table  -> {config.REPORTS_DIR / 'model_scores.csv'}")
    print(f"Saved figure -> {config.FIGURE_DIR / 'confusion_matrices.png'}")
    return table


if __name__ == "__main__":
    evaluate()
