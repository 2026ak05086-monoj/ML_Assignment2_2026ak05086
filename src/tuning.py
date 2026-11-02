"""Cross-validated hyperparameter tuning.

Each ``tune_*`` routine searches a small, sensible grid for one classifier with
stratified 5-fold CV (optimising macro-F1 because the classes are imbalanced),
refits the best pipeline on the whole training split and persists it under
``model/``. These are the models the evaluation step and the Streamlit app use.

    python -m src.tuning logistic_regression
"""
from __future__ import annotations

import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config
from src.features import make_split
from src.modeling import build_pipeline, quick_scores, save_model

# Shared CV splitter so every search is comparable.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
SCORING = "f1_macro"


def tune_logistic_regression():
    """Search regularisation strength and class weighting."""
    base = LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE)
    grid = {
        "classifier__C": [0.05, 0.1, 0.5, 1.0, 5.0],
        "classifier__class_weight": [None, "balanced"],
    }
    return GridSearchCV(build_pipeline(base), grid, scoring=SCORING, cv=CV, n_jobs=-1)


TUNERS = {
    "logistic_regression": tune_logistic_regression,
}


def run_tuning(key: str):
    """Fit the search for ``key``, persist the best pipeline and report it."""
    x_train, x_test, y_train, y_test = make_split()
    search = TUNERS[key]()
    search.fit(x_train, y_train)

    best = search.best_estimator_
    path = save_model(best, key)

    display_name, _ = config.MODEL_REGISTRY[key]
    print(f"[tuned] {display_name}")
    print(f"  best params : {search.best_params_}")
    print(f"  cv {SCORING}: {search.best_score_:.4f}")
    print(f"  test        : {quick_scores(best, x_test, y_test)}")
    print(f"  saved       -> {path}")
    return search, best, path


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "logistic_regression"
    run_tuning(model_key)
