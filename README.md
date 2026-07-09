"""
The Accountant's Eye — Credit Default Risk
An end-to-end ML case study on the HMEQ home-equity dataset.
Storytelling app: EDA → preprocessing → model comparison → feature importance
→ imbalance handling → conclusion → live predictor.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ================================================================= config
st.set_page_config(
    page_title="The Accountant's Eye — Credit Default Risk",
    page_icon="◉",
    layout="wide",
)

MODEL_DIR = Path("model")
FIG_DIR = Path("figures")

# ================================================================= theme
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root{
  --ink:#132132; --ink-soft:#33475e; --paper:#F5F4F0; --card:#FFFFFF;
  --gold:#B8862F; --gold-soft:#F0E4C8; --green:#1F7A4D; --red:#B23A3A;
  --rule:#E4E1D9; --muted:#6B7684;
}
html, body, [class*="css"], .stApp, p, li, div, span{
  font-family:'Inter',sans-serif; color:var(--ink);
}
.stApp{ background:var(--paper); }
.block-container{ padding-top:2.2rem; max-width:1050px; }

h1,h2,h3,h4{ font-family:'Space Grotesk',sans-serif; color:var(--ink);
  letter-spacing:-0.02em; font-weight:700; }
h2{ font-size:1.55rem; margin-top:0.3rem; padding-bottom:.35rem;
  border-bottom:2px solid var(--gold-soft); }
h3{ font-size:1.15rem; color:var(--ink-soft); }

/* numbers everywhere in monospace — the ledger signature */
[data-testid="stMetricValue"]{ font-family:'JetBrains Mono',monospace!important;
  font-weight:700; color:var(--ink); }
[data-testid="stMetricLabel"]{ font-family:'JetBrains Mono',monospace!important;
  text-transform:uppercase; letter-spacing:.05em; font-size:.7rem!important;
  color:var(--muted); }
[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--rule);
  border-left:3px solid var(--gold); border-radius:6px; padding:14px 16px; }

.stDataFrame{ font-family:'JetBrains Mono',monospace; }

/* hero */
.hero{ background:var(--ink); color:#F5F4F0; border-radius:10px;
  padding:34px 40px; margin-bottom:8px; position:relative; overflow:hidden; }
.hero:before{ content:"◉"; position:absolute; right:26px; top:50%;
  transform:translateY(-50%); font-size:120px; color:rgba(184,134,47,.16);
  line-height:1; }
.hero .eyebrow{ color:var(--gold); }
.hero h1{ color:#FCFBF7; font-size:2.5rem; margin:.1rem 0 .3rem; }
.hero .tag{ font-family:'Space Grotesk'; font-size:1.05rem;
  color:#C7CfD8; font-weight:500; max-width:640px; }
.hero .meta{ font-family:'JetBrains Mono',monospace; font-size:.78rem;
  color:#8FA0B2; margin-top:14px; letter-spacing:.02em; }

/* eyebrow label */
.eyebrow{ font-family:'JetBrains Mono',monospace; text-transform:uppercase;
  letter-spacing:.14em; font-size:.72rem; font-weight:500; color:var(--gold);
  margin-bottom:.2rem; }

/* callout cards */
.callout{ background:var(--card); border:1px solid var(--rule);
  border-left:4px solid var(--gold); border-radius:6px; padding:16px 20px;
  margin:14px 0; font-size:.95rem; line-height:1.55; }
.callout.key{ border-left-color:var(--ink); background:#FBFAF7; }
.callout.good{ border-left-color:var(--green); }
.callout.bad{ border-left-color:var(--red); }
.callout b{ color:var(--ink); }
.callout .mono{ font-family:'JetBrains Mono',monospace; font-weight:500; }

/* verdict box */
.verdict{ border-radius:8px; padding:20px 24px; font-family:'Space Grotesk';
  font-weight:600; font-size:1.1rem; }
.verdict.approve{ background:#EAF4EE; border:1px solid var(--green); color:#155C3A; }
.verdict.decline{ background:#F7EBEB; border:1px solid var(--red); color:#8A2B2B; }
.verdict .prob{ font-family:'JetBrains Mono',monospace; font-size:1.6rem; }

/* sidebar */
[data-testid="stSidebar"]{ background:#FBFAF7; border-right:1px solid var(--rule); }
[data-testid="stSidebar"] .eyebrow{ padding:0 4px; }

hr{ border-color:var(--rule); }
a{ color:var(--gold); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def callout(html, kind=""):
    st.markdown(f'<div class="callout {kind}">{html}</div>', unsafe_allow_html=True)


# ================================================================= loaders
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

# ================================================================= hero
st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Credit default risk · ML case study</div>
  <h1>The Accountant's Eye</h1>
  <div class="tag">A trained eye for default risk — reading what an applicant's
  numbers reveal, and showing exactly why some models see it and others can't.</div>
  <div class="meta">HMEQ · 5,960 applicants · ~20% default · leakage-safe pipeline
  · XGBoost · macro-F1 0.854</div>
</div>
""",
    unsafe_allow_html=True,
)

# ================================================================= nav
SECTIONS = ["Overview", "The data", "EDA & insights", "Preprocessing",
            "Model comparison", "Feature importance", "Handling imbalance",
            "Conclusion", "Live predictor"]
st.sidebar.markdown('<div class="eyebrow">The case study</div>', unsafe_allow_html=True)
section = st.sidebar.radio("Navigate", SECTIONS, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown('<div class="eyebrow">Headline</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    "**XGBoost** — macro-F1 `0.854`, PR-AUC `0.859`, catches **81%** of defaulters."
)
st.sidebar.caption("A demonstration on public data, not a real lending tool.")

st.write("")


# ================================================================= OVERVIEW
if section == "Overview":
    eyebrow("01 — Overview")
    st.header("Predicting who defaults, and understanding why")
    st.markdown(
        "Predicting whether a home-equity loan applicant will **default** is a classic "
        "credit-risk problem — and the two mistakes are not equally costly. **Missing a "
        "defaulter** (approving a loan that goes bad) is far more expensive than a "
        "**false alarm** (declining someone who would have repaid). That asymmetry "
        "shapes every decision in this project."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", "5,960")
    c2.metric("Default rate", "19.9%")
    c3.metric("Best macro-F1", "0.854")
    c4.metric("Defaulters caught", "81%")

    st.subheader("The through-line")
    callout(
        "The data's risk is <b>threshold-shaped</b> — flat, then a sharp jump. That one "
        "fact explains everything downstream: which metric to trust, why imputation "
        "beats dropping, and above all <b>why tree models win and linear models "
        "can't</b>. This app follows that thread end to end.",
        "key",
    )
    st.markdown(
        "- **Data audit** → the ~20% imbalance disqualifies accuracy; the missing-data "
        "pattern forces imputation over dropping.\n"
        "- **EDA** → default rate jumps at thresholds (any derogatory mark ~triples "
        "risk) — the central non-linearity.\n"
        "- **Model comparison** → five models form a clean *flexibility ladder* that "
        "maps onto that non-linearity.\n"
        "- **Feature importance** → a model-free vs model-based *gap* proves the tree "
        "extracts interaction signal the others are blind to.\n"
        "- **Threshold** → the decision cutoff is a business trade-off, exposed live."
    )
    callout(
        "Every feature the model uses is known at <b>application time</b> — no "
        "post-outcome information leaks in. That discipline is what makes the results "
        "trustworthy.",
    )


# ================================================================= THE DATA
elif section == "The data":
    eyebrow("02 — The data")
    st.header("HMEQ home-equity applicants")
    st.markdown(
        "Twelve features plus the target `BAD` (1 = defaulted / seriously delinquent, "
        "0 = repaid). Here is what each column means:"
    )
    dd = pd.DataFrame([{"Column": k, "Description": v} for k, v in descriptions.items()])
    st.dataframe(dd, use_container_width=True, hide_index=True)

    st.subheader("Target balance")
    col1, col2 = st.columns([1, 1])
    with col1:
        if (FIG_DIR / "target_dist.png").exists():
            st.image(str(FIG_DIR / "target_dist.png"))
    with col2:
        callout(
            "About <b>20%</b> of applicants defaulted. This is why <b>accuracy is "
            "misleading</b> — a model predicting <i>'good'</i> for everyone scores 80% "
            "accuracy while catching zero defaulters. The headline metrics are "
            "<b>macro-F1</b> (weights both classes equally) and <b>PR-AUC</b> (how well "
            "the model ranks the rare defaulters).",
        )


# ================================================================= EDA
elif section == "EDA & insights":
    eyebrow("03 — Exploratory analysis")
    st.header("What the eye learns to look for")

    st.subheader("Data-quality audit")
    callout(
        "Missing data is everywhere — <span class='mono'>DEBTINC 21.3%</span>, "
        "<span class='mono'>DEROG 11.9%</span>, <span class='mono'>DELINQ 9.7%</span>, "
        "others 2–9%; only <span class='mono'>BAD</span> and <span class='mono'>LOAN</span> "
        "complete. Because gaps sit in <i>different rows</i> across columns, dropping any "
        "row with a null would lose ~half the data → the decision is <b>imputation</b> "
        "(validated later). Categoricals: <span class='mono'>JOB</span>'s Self &amp; "
        "Sales groups are small, so their risk rates are noisier.",
    )

    st.subheader("Risk relationships are non-linear — the key finding")
    if (FIG_DIR / "nonlinearity.png").exists():
        st.image(str(FIG_DIR / "nonlinearity.png"))
    callout(
        "Default rate does <b>not</b> rise in a straight line — it jumps at thresholds:"
        "<br>• <b>DELINQ</b>: ~16% → ~45% → ~65% — risk <i>accelerates</i>."
        "<br>• <b>DEROG</b>: ~18% → ~60% at the <i>first</i> derogatory mark — a near "
        "<b>step function</b>."
        "<br>• <b>DEBTINC</b>: flat ~5–8%, then <b>spikes to ~27%</b> at the highest "
        "debt-to-income."
        "<br>• <b>CLAGE</b>: smooth decline 33% → 10% — longer credit history, lower risk.",
        "key",
    )
    st.markdown(
        "**Why this decides the model choice:** a linear model fits one constant "
        "coefficient per feature — it *cannot* represent 'flat, then a cliff.' Tree "
        "ensembles split exactly at these thresholds (`DELINQ ≥ 2?`, `any DEROG?`). "
        "This is the mechanism behind the entire performance gap you'll see."
    )

    st.subheader("Concrete signals")
    st.markdown(
        "- Defaulters carry higher **debt-to-income** (median `38.1` vs `34.5`).\n"
        "- Defaulters have **shorter credit histories** (`133` vs `180` months).\n"
        "- Default rate varies sharply by **occupation** — Sales `34.9%` and "
        "Self-employed `30.1%` vs Office `13.2%`.\n"
        "- `DELINQ`/`DEROG` medians are `0` for both classes — a *median artifact*, "
        "not 'no signal'; the signal is in the non-zero tail (see the plots above)."
    )

    st.subheader("Feature correlation")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        if (FIG_DIR / "corr.png").exists():
            st.image(str(FIG_DIR / "corr.png"))
    with col2:
        callout(
            "Only one strong pair: <b>MORTDUE ↔ VALUE = 0.88</b> (mortgage owed tracks "
            "property value). This would destabilise a <i>linear</i> model's "
            "coefficients — but the final model is tree-based, which is "
            "<b>indifferent</b> to multicollinearity, so no action is needed. It does "
            "explain why permutation importance later under-credits both features.",
        )


# ================================================================= PREPROCESSING
elif section == "Preprocessing":
    eyebrow("04 — Preprocessing")
    st.header("A leakage-safe pipeline, decided on evidence")

    st.subheader("Impute, don't drop — and here's the proof")
    c1, c2 = st.columns([1, 1])
    c1.metric("Impute (keep rows) · CV macro-F1", "0.690")
    c2.metric("Drop <5%-null rows · CV macro-F1", "0.686")
    callout(
        "The two approaches were compared on <b>cross-validated macro-F1</b> (training "
        "data only — the test set stays untouched). <b>Imputation won</b> — higher "
        "score <i>and</i> it keeps all the data. A neat lesson: four columns each under "
        "5% null still dropped <b>9.8%</b> of rows, because their gaps fall in "
        "<i>different</i> records.",
        "key",
    )

    st.subheader("The pipeline")
    st.markdown(
        "All preprocessing lives *inside* a scikit-learn pipeline, so imputation "
        "medians, the scaler, and the encoder are fit on **training folds only** and "
        "never see the test data:\n"
        "- **Numeric** → median imputation → standard scaling\n"
        "- **Categorical** (REASON, JOB) → most-frequent imputation → one-hot encoding\n\n"
        "The split is **stratified random** (keeping ~20% defaults in both sets). "
        "Unlike a time-series problem, these are independent applicants with no "
        "chronology — so a random split is correct, and being able to say *why* the "
        "rule differs is itself a signal of understanding."
    )


# ================================================================= MODELS
elif section == "Model comparison":
    eyebrow("05 — Model comparison")
    st.header("Five models, one flexibility ladder")
    st.markdown(
        "Every model trained in the same leakage-safe pipeline, evaluated on the same "
        "held-out test set. Sorted by macro-F1:"
    )

    mdf = pd.DataFrame(results["models"])
    st.dataframe(mdf, use_container_width=True, hide_index=True)

    callout(
        "The results form a clean ladder that maps exactly onto how flexibly each model "
        "can bend to the data — and the data's risk is <b>threshold-shaped</b>:"
        "<br>• <b>Linear models</b> (logistic regression, linear SVM) plateau at "
        "<span class='mono'>macro-F1 ≈ 0.67–0.69</span> — one constant slope per "
        "feature can't encode 'safe until a threshold, then sharply risky.'"
        "<br>• <b>RBF SVM lands in the middle</b> "
        "(<span class='mono'>0.777</span>) — its kernel bends the boundary (beating the "
        "linear models by ~0.09) but its <i>smooth</i> curves can't match the "
        "<i>step-functions</i> trees carve (trailing XGBoost by ~0.08)."
        "<br>• <b>XGBoost tops every metric</b> "
        "(<span class='mono'>0.854</span>) — it splits precisely at thresholds and "
        "models interactions natively.",
        "key",
    )
    st.markdown(
        "A textbook **linear < smooth-nonlinear < step-nonlinear** ordering. The gap is "
        "even wider on **PR-AUC** (`0.859` vs `0.710` vs `~0.57`) — on the metric that "
        "matters most for ranking rare defaulters, the tree's edge is larger still."
    )

    st.subheader("The business view — who gets caught")
    cm = results["confusion_xgb"]
    cm_df = pd.DataFrame(cm, index=["Actually good", "Actually bad"],
                         columns=["Predicted good", "Predicted bad"])
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.dataframe(cm_df, use_container_width=True)
    with col2:
        total_bad = cm[1][0] + cm[1][1]
        flagged = cm[0][1] + cm[1][1]
        callout(
            f"Of <b>{total_bad} real defaulters</b>, the model catches "
            f"<b>{cm[1][1]}</b> — about <b>81% recall</b>. Of everyone it flags as bad, "
            f"<b>{cm[1][1]}/{flagged} ≈ 73%</b> truly are. For a lender, that recall is "
            "what protects the loan book.",
        )


# ================================================================= IMPORTANCE
elif section == "Feature importance":
    eyebrow("06 — Feature importance")
    st.header("Two views — and the gap between them")
    st.markdown(
        "A data scientist asks two *different* questions. **Model-free** (mutual "
        "information) scores each feature's raw signal, no model involved. "
        "**Model-based** (permutation on XGBoost) shows what the deployed model actually "
        "uses. The *gap* is the real insight."
    )

    if "importance" in results:
        idf = pd.DataFrame(results["importance"])
        st.dataframe(idf, use_container_width=True, hide_index=True)

    callout(
        "<span class='mono'>rank_gap</span> = model-free rank − model-based rank. "
        "<b>Positive</b> → the model uses a feature <i>more</i> than its standalone "
        "signal justifies (it matters through <b>interactions/thresholds</b>). "
        "<b>Negative</b> → <b>redundant</b>.",
    )
    callout(
        "<b>CLNO is the smoking gun (gap +5).</b> Number of credit lines had "
        "<i>essentially zero</i> linear signal (F-stat ≈ 0.1) and near-bottom mutual "
        "information — yet XGBoost ranks it <b>4th most important</b>. It carries no "
        "standalone signal, only <b>conditional</b> signal (it matters combined with "
        "other features). This is the clearest proof the tree extracts value that "
        "univariate screening and linear models are completely blind to.",
        "key",
    )
    st.markdown(
        "- **`VALUE` and `LOAN` are redundant (gap −6 each)** — they looked important "
        "univariately but the model relegates them; `VALUE`'s signal is absorbed by its "
        "0.88-correlated twin `MORTDUE`.\n"
        "- **`DELINQ`, `CLAGE`, `DEROG` have positive gaps** — the model leans on these "
        "threshold-shaped drivers more than their univariate scores suggest.\n\n"
        "This table is the quantitative echo of the whole project: where the tree uses a "
        "feature *more* than statistics justify, it's capturing non-linearity; where "
        "*less*, that signal was duplicated."
    )


# ================================================================= IMBALANCE
elif section == "Handling imbalance":
    eyebrow("07 — Imbalance & threshold")
    st.header("Class weighting, SMOTE, and the decision cutoff")
    st.markdown(
        "Three strategies compared fairly on the same held-out test set — SMOTE applied "
        "correctly (training folds only, no leakage into the test set)."
    )
    idf = pd.DataFrame(results["imbalance_test"])
    st.dataframe(idf, use_container_width=True, hide_index=True)
    callout(
        "<b>Class weighting (scale_pos_weight) won</b> — and, counter-intuitively, "
        "<b>SMOTE caught <i>fewer</i> defaulters</b> (169 vs 193). Its synthetic, "
        "interpolated samples blurred the decision boundary rather than sharpening it. "
        "SMOTE is not automatic — the simpler method beat it here, reported honestly.",
        "bad",
    )

    st.subheader("Threshold is a business choice")
    callout(
        "The default 0.5 cutoff is arbitrary. Lowering it flags more applicants as risky "
        "— catching more defaulters at the cost of more false alarms. Recall on "
        "defaulters swings from <b>~74%</b> (threshold 0.6) to <b>~90%</b> (threshold "
        "0.3). Because a missed default costs far more than a false decline, a lower "
        "threshold (~0.35) is often the right call — the metric alone doesn't decide, "
        "the cost asymmetry does. Try it live in the predictor.",
        "good",
    )


# ================================================================= CONCLUSION
elif section == "Conclusion":
    eyebrow("08 — Conclusion")
    st.header("Why the models differed")
    callout(
        "<b>Performance here is about inductive bias, not tuning.</b> The winning model "
        "wins because credit-default risk is threshold-shaped and interaction-driven, "
        "and trees are built to carve thresholds and combine features — while a straight "
        "line cannot describe a cliff.",
        "key",
    )
    st.markdown(
        "**1. The data is governed by non-linear, threshold-shaped risk.** Default rate "
        "is flat then jumps — any derogatory mark roughly triples risk, each delinquency "
        "compounds, debt-to-income only bites at the extreme. Everything follows from "
        "this.\n\n"
        "**2. The five-model results form a clean flexibility ladder.** Linear models "
        "(logreg, linear SVM) plateau at ~0.69 — they can't bend to thresholds. RBF SVM "
        "lands at 0.777 — its kernel bends but stays smooth. XGBoost tops out at 0.854 "
        "because its inductive bias *matches the shape of the data*.\n\n"
        "**3. The two importance views prove the mechanism.** The rank-gap shows the "
        "tree using interaction features (`CLNO` +5, `DELINQ` +2, `CLAGE` +2) more than "
        "their univariate signal justifies — with `CLNO` (zero linear signal, 4th "
        "most-used) the smoking gun — while relegating redundant features (`VALUE`, "
        "`LOAN`). Exactly what a model with the right bias should do, and exactly what "
        "linear models and univariate screening cannot."
    )
    st.subheader("Final model")
    st.markdown(
        "**XGBoost** with `scale_pos_weight`, at a business-tuned threshold (~0.35) to "
        "prioritise catching defaulters. SMOTE was tested and lost. Deployed below."
    )


# ================================================================= PREDICTOR
elif section == "Live predictor":
    eyebrow("09 — Live predictor")
    st.header("Run the eye over an applicant")
    st.markdown(
        "Enter an applicant's details and set the decision threshold. The model returns "
        "a **default probability**; the threshold decides the recommendation. Hover the "
        "ⓘ on any field for its meaning."
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

        st.write("")
        if high_risk:
            st.markdown(
                f'<div class="verdict decline">▲ Flagged high-risk &nbsp;·&nbsp; '
                f'default probability <span class="prob">{prob:.1%}</span><br>'
                f'<span style="font-weight:400;font-size:.95rem">At or above the '
                f'{threshold:.0%} threshold — recommend manual review or decline.</span>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="verdict approve">● Within tolerance &nbsp;·&nbsp; '
                f'default probability <span class="prob">{prob:.1%}</span><br>'
                f'<span style="font-weight:400;font-size:.95rem">Below the '
                f'{threshold:.0%} threshold — recommend approve.</span>'
                f'</div>', unsafe_allow_html=True)
        st.caption(
            "Demonstration model on a public dataset — not a real lending decision.")