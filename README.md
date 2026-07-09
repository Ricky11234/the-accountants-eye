# The Accountant's Eye — Credit Default Risk

> *A trained eye for default risk — reading what an applicant's numbers reveal,
> and showing exactly why some models see it and others can't.*

An end-to-end, leakage-safe machine-learning case study on the **HMEQ** home-equity
loan dataset (5,960 applicants, ~20% default rate). It runs from raw data through EDA,
preprocessing decisions, a five-model comparison, a two-view feature-importance
analysis, imbalance handling, and a deployed interactive predictor.

**Live app:** _(add your Streamlit Cloud URL here)_

## Headline result — five models on a held-out test set

| Model | macro-F1 | accuracy | PR-AUC | F1 (default) |
|---|---|---|---|---|
| **XGBoost (scale_pos_weight)** | **0.854** | 0.903 | **0.859** | **0.769** |
| SVM — RBF kernel (balanced) | 0.777 | 0.846 | 0.710 | 0.653 |
| Logistic Regression (no weights) | 0.686 | 0.841 | 0.569 | 0.465 |
| SVM — linear (balanced) | 0.675 | 0.766 | 0.570 | 0.503 |
| Logistic Regression (balanced) | 0.669 | 0.757 | 0.570 | 0.498 |

XGBoost catches **81%** of real defaulters (recall on the default class).

## The core finding — a flexibility ladder

The five models line up as **linear < smooth-nonlinear < step-nonlinear**, which maps
exactly onto the data's structure:

- **Linear models** (logistic regression, linear SVM) plateau at ~0.69 — one constant
  slope per feature can't encode "safe until a threshold, then sharply risky."
- **RBF SVM** lands in the middle (0.777) — its kernel bends the boundary but produces
  *smooth* curves, not the *step functions* the data has.
- **XGBoost** tops out at 0.854 — it splits precisely at thresholds and models
  interactions natively. Its inductive bias **matches the shape of the data**.

EDA shows *why*: default rate is threshold-shaped — any derogatory mark roughly triples
risk, each delinquency compounds, debt-to-income only bites at the extreme.

## The standout inference — `CLNO`, a pure interaction feature

A two-view importance analysis (model-free mutual information vs model-based permutation
importance) surfaced something EDA alone could not. **`CLNO` (number of credit lines) has
essentially zero linear signal — an F-stat of ~0.1 — yet XGBoost ranks it the 4th most
important feature** (the largest positive rank-gap, +5). It carries no *standalone*
signal, only *conditional* signal: it matters only in combination with other features.
This is the clearest evidence in the project that the tree extracts interaction structure
that univariate screening and linear models are completely blind to. Conversely, `VALUE`
and `LOAN` (rank-gap −6) look important univariately but are redundant with correlated
twins, so the model relegates them.

## What makes this project rigorous

- **No leakage** — every feature is application-time; preprocessing is fit on training
  folds only, inside a scikit-learn pipeline.
- **Evidence-based preprocessing** — imputation vs row-dropping decided by cross-validated
  macro-F1 (0.690 vs 0.686), not by habit.
- **Explained model choice** — the model comparison and importance analysis *demonstrate*
  the non-linearity mechanism, not just assert it.
- **Honest imbalance handling** — SMOTE was tested and *lost* to class weighting (caught
  fewer defaulters); reported honestly.
- **Business-aware threshold** — the decision cutoff is framed as a recall/precision
  trade-off and exposed as a live slider.

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
│   ├── results.json                  # 5-model comparison, importance, imbalance test
│   └── column_descriptions.json
├── figures/               # EDA plots rendered in the app
└── notebooks/
    └── credit_risk.ipynb  # the full analysis
```

## Data

HMEQ (Home Equity) — a public credit-risk dataset. All features are application-time
attributes (loan amount, property value, debt-to-income, delinquency and derogatory
counts, credit-history age, occupation, etc.).

*This is a demonstration project on public data, not a real lending tool.*