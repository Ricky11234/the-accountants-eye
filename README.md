# The Accountant's Eye — Credit Default Risk

> *A trained eye for default risk — reading what an applicant's numbers reveal.*

An end-to-end, leakage-safe machine-learning project predicting home-equity loan
default on the **HMEQ** dataset (5,960 applicants, ~20% default rate). The project
runs from raw data through EDA, preprocessing decisions, model comparison, and a
deployed interactive predictor.

**Live app:** _(add your Streamlit Cloud URL here after deploy)_

## Headline result

| Model | macro-F1 | accuracy | PR-AUC | F1 (default class) |
|---|---|---|---|---|
| Logistic Regression (no weights) | 0.686 | 0.841 | 0.569 | 0.465 |
| Logistic Regression (balanced) | 0.669 | 0.757 | 0.570 | 0.498 |
| **XGBoost (scale_pos_weight)** | **0.854** | **0.903** | **0.859** | **0.769** |

XGBoost catches **81%** of real defaulters (recall on the default class).

## What makes this project rigorous

- **No leakage** — every feature is known at application time; preprocessing is fit
  on training folds only, inside a scikit-learn pipeline.
- **Evidence-based preprocessing** — imputation vs row-dropping was decided by a
  cross-validated comparison, not by habit.
- **Explained model choice** — the EDA shows risk relationships are *non-linear*
  (sharp threshold jumps in delinquency, derogatory reports, and debt-to-income),
  which is precisely why the tree-based XGBoost beats linear logistic regression.
- **Honest imbalance handling** — SMOTE was tested and *lost* to class weighting
  (it caught fewer defaulters); the simpler method is reported as the winner.
- **Business-aware threshold** — the decision cutoff is framed as a
  recall/precision trade-off and exposed as a live slider in the app.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo layout

```
├── app.py                 # Streamlit storytelling app + live predictor
├── requirements.txt       # versions pinned to the training environment
├── model/
│   ├── credit_risk_pipeline.joblib   # full pipeline: preprocessing + XGBoost
│   ├── feature_info.json
│   ├── results.json
│   └── column_descriptions.json
├── figures/               # EDA plots rendered in the app
└── notebooks/             # the analysis notebook
```

## Data

HMEQ (Home Equity) dataset — a public credit-risk teaching dataset. All features
are application-time attributes (loan amount, property value, debt-to-income,
delinquency and derogatory counts, credit-history age, occupation, etc.).

*This is a demonstration project on public data, not a real lending tool.*
