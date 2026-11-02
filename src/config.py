"""Central configuration for the student-outcome classification project.

Every path, column name and the random seed live here so the rest of the
pipeline stays reproducible and there is a single place to change things.
"""
from __future__ import annotations

from pathlib import Path

# --- Reproducibility -------------------------------------------------------
# One seed reused for the split and for every estimator that accepts it.
RANDOM_STATE = 42

# --- Project layout --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"

# CSV the evaluator uploads in the Streamlit app (held-out test split only).
TEST_DATA_FILE = PROJECT_ROOT / "test_data.csv"

# --- Dataset ---------------------------------------------------------------
# UCI ML Repository #697 - "Predict Students' Dropout and Academic Success".
UCI_DATASET_ID = 697
UCI_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
RAW_CSV = RAW_DIR / "student_outcomes.csv"

# The label column and the order we want classes reported in.
TARGET_COLUMN = "Target"
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]

# --- Train / test split ----------------------------------------------------
TEST_SIZE = 0.20

# --- Model registry --------------------------------------------------------
# Logical key -> (display label, saved pipeline filename). Filled in as the
# five classifiers get implemented in later stages.
MODEL_REGISTRY = {
    "logistic_regression": ("Logistic Regression", "logistic_regression.joblib"),
    "decision_tree": ("Decision Tree", "decision_tree.joblib"),
    "knn": ("K-Nearest Neighbors", "knn.joblib"),
    "naive_bayes": ("Naive Bayes (Gaussian)", "naive_bayes.joblib"),
    "svm": ("Support Vector Machine", "svm.joblib"),
}


def ensure_directories() -> None:
    """Create the data / model / report folders if they do not exist yet."""
    for folder in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
