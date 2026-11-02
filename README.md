# Student Academic Outcome Classifier

An end-to-end machine learning project that predicts whether a higher-education
student will **drop out**, remain **enrolled**, or **graduate**, based on
enrolment, demographic and first/second-semester academic records. Five
classifiers are trained and compared, and the saved models are served through
an interactive Streamlit app that scores an uploaded test set without
retraining.

> Machine Learning (AIML) — Assignment 2. The project is built up in small,
> reviewable stages; the commit history shows how each piece was added.

## Build status

- [x] Stage 1 — Project scaffold & environment
- [x] Stage 2 — Data acquisition & exploration
- [x] Stage 3 — Preprocessing & train/test split
- [x] Stage 4 — Logistic Regression (baseline + tuned)
- [ ] Stage 5 — Decision Tree (baseline + tuned)
- [ ] Stage 6 — K-Nearest Neighbors (baseline + tuned)
- [ ] Stage 7 — Naive Bayes (baseline + tuned)
- [ ] Stage 8 — Support Vector Machine (baseline + tuned)
- [ ] Stage 9 — Evaluation & model comparison
- [ ] Stage 10 — Streamlit app
- [ ] Stage 11 — Documentation & deployment

## Project layout

```
2/
├── app.py                 # Streamlit evaluation dashboard (added in Stage 7)
├── requirements.txt       # runtime dependencies
├── README.md              # this file
├── test_data.csv          # held-out test split: input features + target
├── model/                 # five saved scikit-learn pipelines (.joblib)
├── reports/               # metrics table and generated figures
└── src/
    ├── config.py          # paths, seed, dataset + model registry
    ├── dataset.py         # download & load the raw UCI data
    ├── eda.py             # exploratory data analysis
    ├── features.py        # preprocessing pipeline + train/test split
    ├── modeling.py        # build & train the five classifiers
    └── evaluation.py      # evaluation metrics & model comparison
```

## Dataset

Predict Students' Dropout and Academic Success — UCI Machine Learning
Repository (ID 697). 4,424 instances, 36 input features, 3 target classes
(`Dropout`, `Enrolled`, `Graduate`).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

_Detailed usage, the model comparison table and observations are added as the
project progresses (Stage 8)._
