"""Model definitions and a small training harness.

Every classifier is wrapped in a pipeline that first applies the shared
preprocessor (see :mod:`src.features`) and then the estimator, so a saved model
is fully self-contained: it accepts the raw feature columns and performs the
encoding/scaling internally. That is exactly what the Streamlit app relies on –
it never has to know how the data was prepared.

Baseline estimators live here; the cross-validated search that produces the
final persisted models lives in :mod:`src.tuning`.

    python -m src.modeling logistic_regression
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.features import build_preprocessor, make_split


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator behind the shared preprocessing step."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def quick_scores(pipeline: Pipeline, features, target) -> dict[str, float]:
    """Lightweight accuracy / macro-F1 used for sanity checks while training."""
    predicted = pipeline.predict(features)
    return {
        "accuracy": round(accuracy_score(target, predicted), 4),
        "macro_f1": round(f1_score(target, predicted, average="macro"), 4),
    }


def save_model(pipeline: Pipeline, key: str) -> Path:
    """Persist a trained pipeline under ``model/`` using its registry name."""
    config.ensure_directories()
    _, filename = config.MODEL_REGISTRY[key]
    destination = config.MODEL_DIR / filename
    joblib.dump(pipeline, destination)
    return destination


# --- Baseline estimators ---------------------------------------------------
# Deliberately simple starting points; tuning.py searches around these.

def logistic_regression_baseline() -> LogisticRegression:
    """Plain multinomial logistic regression (lbfgs, no class weighting)."""
    return LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)


def decision_tree_baseline() -> DecisionTreeClassifier:
    """Unrestricted CART tree - expected to overfit heavily before tuning."""
    return DecisionTreeClassifier(random_state=config.RANDOM_STATE)


def knn_baseline() -> KNeighborsClassifier:
    """Default k-NN (k=5, uniform weights) over the scaled feature space."""
    return KNeighborsClassifier()


def naive_bayes_baseline() -> GaussianNB:
    """Gaussian Naive Bayes with library defaults (var_smoothing=1e-9)."""
    return GaussianNB()


BASELINE_BUILDERS = {
    "logistic_regression": logistic_regression_baseline,
    "decision_tree": decision_tree_baseline,
    "knn": knn_baseline,
    "naive_bayes": naive_bayes_baseline,
}


def fit_baseline(key: str):
    """Train a model's baseline estimator; return the pipeline and scores."""
    x_train, x_test, y_train, y_test = make_split()
    pipeline = build_pipeline(BASELINE_BUILDERS[key]())
    pipeline.fit(x_train, y_train)
    scores = {
        "train": quick_scores(pipeline, x_train, y_train),
        "test": quick_scores(pipeline, x_test, y_test),
    }
    return pipeline, scores


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "logistic_regression"
    _, report = fit_baseline(model_key)
    display_name, _ = config.MODEL_REGISTRY[model_key]
    print(f"[baseline] {display_name}")
    print(f"  train: {report['train']}")
    print(f"  test : {report['test']}")
