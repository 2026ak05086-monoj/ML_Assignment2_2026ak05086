# Machine Learning — Assignment 2
# Student Academic Outcome Classifier

An end-to-end machine learning project that predicts whether a higher-education
student will **drop out**, remain **enrolled**, or **graduate**, based on
enrolment, demographic and first/second-semester academic records. Five
classifiers are trained and compared, and the saved models are served through
an interactive Streamlit app that scores an uploaded test set without
retraining.


---

## a. Problem statement

Student attrition is a costly problem for universities and a life-changing one
for students. The goal of this project is to predict a student's **academic
outcome** — `Dropout`, `Enrolled` (still studying), or `Graduate` — at the end
of the normal duration of their course, using information that is available at
enrolment together with first- and second-semester performance. Framed as a
**multi-class classification** problem, an accurate model lets an institution
flag at-risk students early enough to offer support. Because the three outcomes
are imbalanced, the project is evaluated with imbalance-aware metrics (AUC, F1
and MCC) rather than accuracy alone.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — *Predict Students' Dropout and
  Academic Success* (dataset ID 697).
  <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>
- **Instances:** 4,424 &nbsp;|&nbsp; **Features:** 36 &nbsp;|&nbsp; **Target classes:** 3
- **Data quality:** no missing values and no duplicate rows.
- **Target balance:** `Graduate` 2,209 (49.9%), `Dropout` 1,421 (32.1%),
  `Enrolled` 794 (17.9%) — a moderately imbalanced, three-class target.

The 36 features split into two natural groups that drive the preprocessing:

| Group | Count | Examples |
|-------|-------|----------|
| Nominal categoricals (integer-coded) | 17 | Marital status, Application mode, Course, Nationality, Mother's/Father's qualification & occupation, Gender, Scholarship holder, Debtor, Tuition fees up to date |
| Numeric (continuous / counts) | 19 | Admission grade, Age at enrollment, Curricular units 1st & 2nd sem (enrolled/approved/grade…), Unemployment rate, Inflation rate, GDP |

The strongest signal comes from the **approved curricular units** in the first
two semesters, which separate graduates from dropouts clearly (see
`reports/figures/key_features_by_class.png`).

## c. GitHub Repository Link

> **Repository:** <https://github.com/2026ak05086-monoj/ML_Assignment2_2026ak05086>

The repository contains the complete source code, `requirements.txt`, this
`README.md`, the held-out `test_data.csv`, and the five trained pipelines under
`model/`.

## d. Models used

All five classifiers were trained on the **same** 80/20 stratified split of the
dataset. Each model is a scikit-learn `Pipeline` that embeds the shared
preprocessing (one-hot encoding of categoricals + standardisation of numerics),
so the saved model consumes the raw feature columns directly. Hyperparameters
were selected with **5-fold stratified cross-validation optimising macro-F1**
(chosen because the classes are imbalanced).

Metrics below are computed on the held-out test set (885 rows). Precision,
Recall, F1 and AUC use **weighted** averaging; AUC is one-vs-rest.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|:--------:|:---:|:---------:|:------:|:--:|:---:|
| Logistic Regression | **0.7379** | 0.9018 | **0.7797** | **0.7379** | **0.7510** | **0.5956** |
| Decision Tree | 0.7333 | 0.8727 | 0.7186 | 0.7333 | 0.7237 | 0.5584 |
| KNN | 0.7130 | 0.8566 | 0.7022 | 0.7130 | 0.6879 | 0.5215 |
| Naive Bayes | 0.6904 | 0.8369 | 0.6905 | 0.6904 | 0.6843 | 0.4891 |
| Support Vector Machine | 0.7311 | **0.9024** | 0.7637 | 0.7311 | 0.7417 | 0.5784 |

Confusion matrices for every model are in
`reports/figures/confusion_matrices.png`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | **Best overall**, leading on five of the six metrics. The engineered feature space is close to linearly separable, so a regularised linear boundary (`C=0.1`, `class_weight="balanced"`) generalises best and recovers the minority *Enrolled* class far better than the tree/distance models (106/159 correct). AUC of 0.90 shows well-ranked probabilities. |
| Decision Tree | Competitive accuracy (0.733) but the lowest-but-one AUC. The unrestricted baseline overfit perfectly (train F1 = 1.00); capping `max_depth=8` and `min_samples_leaf=20` restored generalisation. Predicts *Dropout* well but a single tree is less stable than the linear/kernel models. |
| KNN | Weakest F1/MCC of the five. In the 244-dimensional one-hot space distances become noisy (curse of dimensionality); even with `k=15` it largely collapses *Enrolled* into *Graduate* (only 40/159 correct). Good at the majority class, poor at the middle class. |
| Naive Bayes | Lowest accuracy and AUC. Its feature-independence and per-feature Gaussian assumptions are violated by the one-hot binary columns — the default model collapsed to 24% accuracy, and only a large variance floor (`var_smoothing=0.1`) made it usable. A fast, simple baseline rather than a contender. |
| Support Vector Machine | A very close second: the highest AUC (0.9024) and strong weighted-F1 (0.742). The RBF kernel with balanced weights handles the minority class almost as well as Logistic Regression, but it is marginally behind on accuracy/F1/MCC and far more expensive to train (probability calibration). |
| **Overall, winner for this dataset** | **Logistic Regression** — it tops Accuracy, Precision, Recall, F1 and MCC, trains in a fraction of the time of the SVM, and is the most interpretable. The SVM is the runner-up and the only model that edges it on AUC (by 0.0006). |

---


## Streamlit app

The app (`app.py`) loads the saved pipelines and lets you:

- Upload a test CSV (the committed `test_data.csv`),
- Pick a model from a dropdown,
- View the six evaluation metrics, and
- Inspect the confusion matrix / classification report

Run it locally with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Live app:** <https://mlassignment-22026ak05086.streamlit.app/>


