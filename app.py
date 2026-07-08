"""
Credit Default Risk — an end-to-end ML case study on the HMEQ home-equity dataset.
Storytelling app: walks through EDA, preprocessing decisions, model comparison,
and ends with an interactive risk predictor.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ config
st.set_page_config(
    page_title="The Accountant's Eye — Credit Default Risk",
    page_icon="◉",
    layout="wide",
)

MODEL_DIR = Path("model")
FIG_DIR = Path("figures")


# ------------------------------------------------------------------ loaders
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_DIR / "credit_risk_pipeline.joblib")


@st.cache_data
def load_json(name):
    with open(MODEL_DIR / name) as f:
        return json.load(f)


pipeline = load_pipeline()
feature_info = load_json("feature_info.json")
results = load_json("results.json")
descriptions = load_json("column_descriptions.json")


# ------------------------------------------------------------------ header
st.title("The Accountant's Eye")
st.markdown(
    "#### A trained eye for default risk — reading what an applicant's numbers reveal."
)
st.markdown(
    "**A leakage-safe, end-to-end machine-learning case study on the HMEQ "
    "home-equity loan dataset.** From raw data and EDA through preprocessing "
    "decisions, model comparison, and a live risk predictor."
)
st.caption(
    "Dataset: 5,960 home-equity loan applicants · Target: default (BAD) · "
    "~20% default rate · Model: XGBoost"
)
st.divider()

# ------------------------------------------------------------------ nav
section = st.sidebar.radio(
    "Jump to",
    ["Overview", "The data", "EDA & insights", "Preprocessing decisions",
     "Model comparison", "Handling imbalance", "Live predictor"],
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Headline result**  \nXGBoost — macro-F1 **0.854**, PR-AUC **0.859**, "
    "catches **81%** of defaulters."
)


# ============================================================= OVERVIEW
if section == "Overview":
    st.header("What this project does")
    st.markdown(
        "Predicting whether a home-equity loan applicant will **default** is a "
        "classic credit-risk problem. It matters because the two mistakes are not "
        "equally costly: **missing a defaulter** (approving a loan that goes bad) "
        "is far more expensive than a **false alarm** (declining an applicant who "
        "would have repaid). That asymmetry shapes every choice below."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Applicants", "5,960")
    c2.metric("Default rate", "~20%")
    c3.metric("Best macro-F1", "0.854")

    st.subheader("How the analysis was built")
    st.markdown(
        "- **Data audit** — confirmed the target imbalance and mapped missing data "
        "across every column before touching it.\n"
        "- **EDA** — found the *non-linear* risk relationships that decide the "
        "model choice.\n"
        "- **Preprocessing** — chose imputation over row-dropping, backed by a "
        "cross-validated comparison, inside a leakage-safe pipeline.\n"
        "- **Modeling** — compared logistic regression against XGBoost, and "
        "explained *why* the tree model wins.\n"
        "- **Imbalance** — tested SMOTE against class weighting and reported the "
        "honest result.\n"
        "- **Threshold** — framed the decision cutoff as a business trade-off, "
        "exposed live in the predictor."
    )
    st.info(
        "Every feature used by the model is known at **application time** — no "
        "post-outcome information leaks in. That discipline is what makes the "
        "results trustworthy.",
        icon="▚",
    )


# ============================================================= THE DATA
elif section == "The data":
    st.header("The data")
    st.markdown(
        "The HMEQ dataset describes home-equity loan applicants. Twelve features "
        "plus the target. Here is what each column means:"
    )
    dd = pd.DataFrame(
        [{"Column": k, "Description": v} for k, v in descriptions.items()]
    )
    st.dataframe(dd, use_container_width=True, hide_index=True)

    st.subheader("Target balance")
    col1, col2 = st.columns([1, 1])
    with col1:
        if (FIG_DIR / "target_dist.png").exists():
            st.image(str(FIG_DIR / "target_dist.png"))
    with col2:
        st.markdown(
            "About **20%** of applicants defaulted. This imbalance is the reason "
            "**accuracy alone is misleading** — a model that predicts *'good'* for "
            "everyone scores 80% accuracy while catching zero defaulters. So the "
            "headline metrics here are **macro-F1** (weights both classes equally) "
            "and **PR-AUC** (how well the model ranks the rare defaulters)."
        )


# ============================================================= EDA
elif section == "EDA & insights":
    st.header("EDA & insights")
    st.caption("What the eye learns to look for — the patterns that signal risk.")

    st.subheader("Risk relationships are non-linear")
    if (FIG_DIR / "nonlinearity.png").exists():
        st.image(str(FIG_DIR / "nonlinearity.png"))
    st.markdown(
        "Each bar shows the **default rate** within a slice of the feature; the "
        "red line is the overall ~20% rate. The relationships are not straight lines:\n"
        "- **DELINQ** (delinquent credit lines): default rate climbs ~16% → ~45% "
        "→ ~65% — it *accelerates*.\n"
        "- **DEROG** (derogatory reports): a near **step-jump** from ~18% to ~60% "
        "at the first derogatory mark.\n"
        "- **DEBTINC** (debt-to-income): flat around 5–8%, then **spikes to ~27%** "
        "at the highest ratios.\n"
        "- **CLAGE** (credit-history age): a smooth *decline* — longer history, "
        "lower risk.\n\n"
        "These threshold effects are the key to the model comparison: a linear "
        "model assumes each unit of a feature adds a fixed amount of risk, so it "
        "**cannot** capture 'risk triples once debt-to-income crosses a line.' A "
        "tree can."
    )

    st.subheader("A few concrete findings")
    st.markdown(
        "- Defaulters carried higher debt-to-income (median **38.1** vs **34.5**).\n"
        "- Defaulters had **shorter** credit histories (median **133** vs **180** "
        "months).\n"
        "- Default rate varies sharply by occupation — **Sales ~35%** and "
        "**Self-employed ~30%** versus **Office ~13%**."
    )

    st.subheader("Feature correlations")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        if (FIG_DIR / "corr.png").exists():
            st.image(str(FIG_DIR / "corr.png"))
    with col2:
        st.markdown(
            "Only one strong pair: **MORTDUE ↔ VALUE (0.88)** — the mortgage owed "
            "tracks the property's value, which makes sense.\n\n"
            "This multicollinearity would distort a *linear* model's coefficients, "
            "but the final model is tree-based (XGBoost), which is **indifferent to "
            "it** — so no action was needed. The right response to a finding depends "
            "on the model you're shipping."
        )


# ============================================================= PREPROCESSING
elif section == "Preprocessing decisions":
    st.header("Preprocessing decisions")

    st.subheader("Missing data: impute, don't drop")
    st.markdown(
        "Most columns had missing values — from ~2% up to **21%** for debt-to-income. "
        "Dropping every row with *any* missing value would have removed roughly half "
        "the data, because the gaps sit in *different* rows across columns.\n\n"
        "So instead of guessing, the two approaches were **compared on "
        "cross-validated macro-F1** (training data only — the test set stays "
        "untouched):"
    )
    comp = pd.DataFrame({
        "Approach": ["Impute (keep all rows)", "Drop rows with <5%-null columns"],
        "CV macro-F1": ["0.698 ± 0.012", "0.686 ± 0.012"],
        "Data kept": ["100%", "~90% (lost 9.8% of train)"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.markdown(
        "**Imputation won** — higher score *and* keeps all the data. A neat lesson "
        "hides here: four columns each under 5% null still dropped ~10% of rows, "
        "because their gaps fall in different records."
    )

    st.subheader("The leakage-safe pipeline")
    st.markdown(
        "All preprocessing lives *inside* a scikit-learn pipeline, so imputation "
        "medians, the scaler, and the one-hot encoder are fit on **training folds "
        "only** and never see the test data:\n"
        "- **Numeric** → median imputation → standard scaling\n"
        "- **Categorical** (REASON, JOB) → most-frequent imputation → one-hot "
        "encoding\n\n"
        "The data is split with a **stratified random split** (keeping ~20% "
        "defaults in both sets). Unlike a time-series problem, these are "
        "independent applicants with no chronology — so a random split is correct."
    )


# ============================================================= MODELS
elif section == "Model comparison":
    st.header("Model comparison")
    st.markdown("Evaluated on a held-out test set the models never saw during training.")

    mdf = pd.DataFrame(results["models"])
    st.dataframe(mdf, use_container_width=True, hide_index=True)

    st.subheader("Why XGBoost beats logistic regression")
    st.markdown(
        "XGBoost wins decisively — macro-F1 **0.854** vs **0.686**, and it catches "
        "far more defaulters (F1 on the bad class **0.769** vs **0.465**). The "
        "reason traces straight back to the EDA:\n\n"
        "The strongest risk drivers relate to default **non-linearly** — flat "
        "regions with sharp jumps at thresholds (recall the DELINQ, DEROG, and "
        "DEBTINC plots). Logistic regression fits a single straight coefficient per "
        "feature, so it *cannot* represent 'safe until debt-to-income crosses a "
        "line, then sharply risky.' XGBoost is built from decision trees that split "
        "exactly at those thresholds, so it captures the structure the linear model "
        "is blind to."
    )

    st.subheader("The business view — who gets caught")
    cm = results["confusion_xgb"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actually good", "Actually bad"],
        columns=["Predicted good", "Predicted bad"],
    )
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.dataframe(cm_df, use_container_width=True)
    with col2:
        st.markdown(
            f"Of **{cm[1][0] + cm[1][1]} real defaulters**, the model catches "
            f"**{cm[1][1]}** — about **81% recall**. Of everyone it flags as bad, "
            f"**{cm[1][1]}/{cm[0][1] + cm[1][1]} ≈ 73%** truly are. For a lender, "
            "that recall is what protects the loan book."
        )


# ============================================================= IMBALANCE
elif section == "Handling imbalance":
    st.header("Handling imbalance")
    st.markdown(
        "With defaults rare, the model needs help attending to them. Three "
        "strategies were compared **fairly** on the same held-out test set — and "
        "SMOTE was applied correctly (only on training folds, never leaking into "
        "the test set)."
    )
    idf = pd.DataFrame(results["imbalance_test"])
    st.dataframe(idf, use_container_width=True, hide_index=True)
    st.markdown(
        "**Class weighting (`scale_pos_weight`) won** — and, counter-intuitively, "
        "**SMOTE caught *fewer* defaulters** (169 vs 193). Its synthetic, "
        "interpolated samples blurred the decision boundary rather than sharpening "
        "it. The takeaway: SMOTE is not automatic — the simpler method beat it here, "
        "and the honest move is to report that."
    )

    st.subheader("Threshold is a business choice")
    st.markdown(
        "The default 0.5 cutoff is arbitrary. Lowering it flags more applicants as "
        "risky — **catching more defaulters at the cost of more false alarms.** "
        "Because a missed default costs far more than a false decline, a lower "
        "threshold (~0.35) is often the right call. You can move it yourself in the "
        "**Live predictor** and watch the decision change."
    )


# ============================================================= PREDICTOR
elif section == "Live predictor":
    st.header("Live predictor")
    st.caption("Run the eye over an applicant.")
    st.markdown(
        "Enter an applicant's details and set the decision threshold. The model "
        "returns a **default probability**; the threshold decides the "
        "recommendation. Hover the ⓘ on any field for its meaning."
    )

    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        with c1:
            LOAN = st.number_input("Loan amount", 0, 100000, 15000, 500,
                                   help=descriptions["LOAN"])
            MORTDUE = st.number_input("Mortgage due", 0, 400000, 70000, 1000,
                                      help=descriptions["MORTDUE"])
            VALUE = st.number_input("Property value", 0, 900000, 100000, 1000,
                                    help=descriptions["VALUE"])
            YOJ = st.number_input("Years on job", 0.0, 60.0, 7.0, 0.5,
                                  help=descriptions["YOJ"])
        with c2:
            DEROG = st.number_input("Derogatory reports", 0, 20, 0, 1,
                                    help=descriptions["DEROG"])
            DELINQ = st.number_input("Delinquent lines", 0, 20, 0, 1,
                                     help=descriptions["DELINQ"])
            CLAGE = st.number_input("Oldest credit line (months)", 0.0, 1200.0,
                                    180.0, 1.0, help=descriptions["CLAGE"])
            NINQ = st.number_input("Recent inquiries", 0, 20, 1, 1,
                                   help=descriptions["NINQ"])
        with c3:
            CLNO = st.number_input("Number of credit lines", 0, 80, 20, 1,
                                   help=descriptions["CLNO"])
            DEBTINC = st.number_input("Debt-to-income ratio", 0.0, 300.0, 35.0,
                                      0.5, help=descriptions["DEBTINC"])
            REASON = st.selectbox("Loan reason", feature_info["reason_categories"],
                                  help=descriptions["REASON"])
            JOB = st.selectbox("Job", feature_info["job_categories"],
                               help=descriptions["JOB"])

        threshold = st.slider(
            "Decision threshold — flag as high-risk above this probability",
            0.10, 0.90, float(feature_info.get("recommended_threshold", 0.35)), 0.01,
            help="Lower = catch more defaulters but more false alarms. "
                 "Higher = fewer false alarms but miss more defaulters.",
        )
        submitted = st.form_submit_button("Assess risk", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "LOAN": LOAN, "MORTDUE": MORTDUE, "VALUE": VALUE, "REASON": REASON,
            "JOB": JOB, "YOJ": YOJ, "DEROG": DEROG, "DELINQ": DELINQ,
            "CLAGE": CLAGE, "NINQ": NINQ, "CLNO": CLNO, "DEBTINC": DEBTINC,
        }])[feature_info["all_features"]]

        prob = float(pipeline.predict_proba(row)[0, 1])
        high_risk = prob >= threshold

        st.divider()
        c1, c2 = st.columns([1, 1.4])
        c1.metric("Default probability", f"{prob:.1%}")
        with c2:
            if high_risk:
                st.error(
                    f"**Flagged high-risk** — probability {prob:.1%} is at or above "
                    f"the {threshold:.0%} threshold. Recommend manual review / "
                    f"decline.",
                    icon="▲",
                )
            else:
                st.success(
                    f"**Within tolerance** — probability {prob:.1%} is below the "
                    f"{threshold:.0%} threshold. Recommend approve.",
                    icon="▚",
                )
        st.caption(
            "This is a demonstration model on a public dataset, not a real lending "
            "decision tool."
        )
