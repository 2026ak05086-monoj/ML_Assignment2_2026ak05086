"""Source package for the student academic-outcome classifier.

Modules are added stage by stage:
    config      - paths, seed, dataset + model registry
    dataset     - download and load the raw UCI data
    eda         - exploratory data analysis helpers
    features    - preprocessing pipeline and train/test split
    modeling    - build and train the five classifiers
    evaluation  - evaluation metrics and model comparison
"""

__version__ = "0.1.0"
