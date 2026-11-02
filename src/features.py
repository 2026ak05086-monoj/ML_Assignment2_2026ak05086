"""Feature preprocessing and the train/test split.

The 36 input columns fall into two groups:

* **Nominal categoricals** - integer *codes* with no ordinal meaning (course,
  marital status, parents' qualifications/occupations, binary flags). These are
  one-hot encoded.
* **Numeric** - application order, grades, curricular-unit counts, age and the
  economic indicators. These are standardised.

``build_preprocessor`` returns the *unfitted* ColumnTransformer that every model
pipeline reuses, so the encoding/scaling is learnt on the training split only
and then applied consistently at inference time. ``make_split`` gives the same
deterministic, stratified split everywhere it is called.

    python -m src.features
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.dataset import load_raw

# Integer-coded nominal categories -> one-hot encoded.
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Continuous / count / ordinal values -> standardised.
NUMERIC_FEATURES = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Create the unfitted preprocessing transformer shared by all models.

    One-hot output is kept dense (``sparse_output=False``) so that estimators
    such as GaussianNB, which cannot consume sparse matrices, work unchanged.
    ``handle_unknown='ignore'`` protects against category codes that only show
    up in the test split.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def load_feature_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix ``X`` and target vector ``y`` from raw data."""
    frame = load_raw()
    features = frame[FEATURE_COLUMNS].copy()
    target = frame[config.TARGET_COLUMN].copy()
    return features, target


def make_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Deterministic, stratified train/test split reused across the project."""
    features, target = load_feature_frame()
    return train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=target,
    )


def export_test_data() -> pd.DataFrame:
    """Write the held-out test split (features + target) to ``test_data.csv``."""
    config.ensure_directories()
    _, x_test, _, y_test = make_split()
    test_frame = x_test.copy()
    test_frame[config.TARGET_COLUMN] = y_test.to_numpy()
    test_frame.to_csv(config.TEST_DATA_FILE, index=False)
    return test_frame


def _describe_split(label: str, target_part: pd.Series) -> None:
    counts = target_part.value_counts().reindex(config.CLASS_ORDER)
    total = len(target_part)
    spread = ", ".join(
        f"{cls}={int(counts[cls])} ({counts[cls] / total:.0%})"
        for cls in config.CLASS_ORDER
    )
    print(f"{label}: {total} rows | {spread}")


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = make_split()

    # Confirm the preprocessor fits and learn the transformed width.
    transformed = build_preprocessor().fit_transform(x_train)

    test_frame = export_test_data()

    print(
        f"Features: {len(FEATURE_COLUMNS)} "
        f"({len(CATEGORICAL_FEATURES)} categorical + {len(NUMERIC_FEATURES)} numeric)"
    )
    _describe_split("Train", y_train)
    _describe_split("Test ", y_test)
    print(f"Preprocessed train matrix: {transformed.shape}")
    print(f"Wrote {config.TEST_DATA_FILE.name} -> {test_frame.shape[0]} rows, "
          f"{test_frame.shape[1]} columns")
