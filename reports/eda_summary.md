# Exploratory Data Analysis

- **Instances:** 4424
- **Columns:** 37 (36 features + target)
- **Missing cells:** 0
- **Duplicate rows:** 0
- **Column dtypes:** {'int64': 29, 'float64': 7, 'str': 1}

## Target balance

- Dropout: 1421 (32.1%)
- Enrolled: 794 (17.9%)
- Graduate: 2209 (49.9%)

## Notes

- Three-class, moderately imbalanced problem (Enrolled is the minority).
- No missing values or duplicate rows, so cleaning effort is minimal.
- Most integer columns are coded categoricals (marital status, course, parents' qualifications/occupations, binary flags); the grade, economic-indicator and age columns are genuinely continuous.
- Approved curricular units (1st & 2nd semester) separate graduates from dropouts most clearly, so they should be strong predictors.
