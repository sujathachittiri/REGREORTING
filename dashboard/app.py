import streamlit as st
import pandas as pd
import numpy as np
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Intelligent Data Quality Monitoring for Regulatory Reporting",
    layout="wide"
)

st.title("📊 Intelligent Data Quality Monitoring for Regulatory Reporting")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📈 Monitoring Dashboard", "🧠 Explainability (XAI)"])

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "data/processed/final_scored_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# -----------------------------
# Basic derived columns
# -----------------------------
df["FLAG_SOURCE"] = "Clean"
df.loc[(df["RULE_ANOMALY_FLAG"] == 1) & (df["ML_ANOMALY_FLAG"] == 0), "FLAG_SOURCE"] = "Rule"
df.loc[(df["RULE_ANOMALY_FLAG"] == 0) & (df["ML_ANOMALY_FLAG"] == 1), "FLAG_SOURCE"] = "ML"
df.loc[(df["RULE_ANOMALY_FLAG"] == 1) & (df["ML_ANOMALY_FLAG"] == 1), "FLAG_SOURCE"] = "Rule + ML"

# Risk bucket based on ensemble score
q99 = df["ENSEMBLE_SCORE"].quantile(0.99)
q95 = df["ENSEMBLE_SCORE"].quantile(0.95)

def risk_bucket(x):
    if x >= q99:
        return "High"
    elif x >= q95:
        return "Medium"
    else:
        return "Low"

df["RISK_BUCKET"] = df["ENSEMBLE_SCORE"].apply(risk_bucket)

# -----------------------------
# ========== TAB 1 ==========
# -----------------------------
with tab1:

    # -----------------------------
    # KPI Row
    # -----------------------------
    total = len(df)
    rule_cnt = int(df["RULE_ANOMALY_FLAG"].sum())
    ml_cnt = int(df["ML_ANOMALY_FLAG"].sum())
    final_cnt = int(df["FINAL_ANOMALY_FLAG"].sum())
    ml_only = int(((df["RULE_ANOMALY_FLAG"] == 0) & (df["ML_ANOMALY_FLAG"] == 1)).sum())

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Records", total)
    c2.metric("Rule Anomalies", rule_cnt)
    c3.metric("ML Anomalies", ml_cnt)
    c4.metric("Final Anomalies", final_cnt)
    c5.metric("ML-only Anomalies", ml_only)

    st.divider()

    # -----------------------------
    # Distribution charts
    # -----------------------------
    st.subheader("📈 Anomaly Distribution")

    dist_df = df["FLAG_SOURCE"].value_counts().reset_index()
    dist_df.columns = ["Source", "Count"]

    st.bar_chart(dist_df.set_index("Source"))

    st.divider()

    # -----------------------------
    # Top risky records
    # -----------------------------
    st.subheader("🔥 Top High-Risk Records (by ML Ensemble Score)")

    top_n = st.slider("Select Top N Records", min_value=10, max_value=200, value=50, step=10)

    top_df = df.sort_values("ENSEMBLE_SCORE", ascending=False).head(top_n)

    show_cols = [
        "Account_ID",
        "Country_Code",
        "Currency",
        "Counterparty_Type",
        "Product_Type",
        "Exposure_Amount",
        "Risk_Weight",
        "Capital_Requirement",
        "ENSEMBLE_SCORE",
        "FLAG_SOURCE",
        "RISK_BUCKET"
    ]

    existing_cols = [c for c in show_cols if c in top_df.columns]

    st.dataframe(top_df[existing_cols], width='stretch')

    st.divider()

    # -----------------------------
    # Filters
    # -----------------------------
    st.subheader("🔎 Filter & Investigate Records")

    flag_filter = st.multiselect(
        "Filter by Flag Source",
        options=df["FLAG_SOURCE"].unique().tolist(),
        default=df["FLAG_SOURCE"].unique().tolist()
    )

    risk_filter = st.multiselect(
        "Filter by Risk Bucket",
        options=df["RISK_BUCKET"].unique().tolist(),
        default=df["RISK_BUCKET"].unique().tolist()
    )

    filtered = df[
        (df["FLAG_SOURCE"].isin(flag_filter)) &
        (df["RISK_BUCKET"].isin(risk_filter))
    ]

    st.write("Filtered rows:", len(filtered))

    st.dataframe(
        filtered.sort_values("ENSEMBLE_SCORE", ascending=False)[existing_cols],
        width='stretch'
    )

    # -----------------------------
    # Explanation text
    # -----------------------------
    st.divider()
    st.subheader("ℹ️ How to Interpret This Dashboard")

    st.markdown("""
    **Rule Anomalies**  
    Detected using deterministic regulatory and data quality rules.

    **ML Anomalies**  
    Detected using an ensemble of unsupervised models (Isolation Forest, Autoencoder, PCA, LOF, OCSVM).

    **ML-only Anomalies**  
    Records not caught by any rule but flagged by ML based on multivariate patterns.

    **Risk Buckets**  
    Based on ML Ensemble Score percentile:
    - High = Top 1%
    - Medium = 95–99 percentile
    - Low = Remaining
    """)

    st.success("✅ Dashboard loaded successfully from final_scored_data.csv")

# -----------------------------
# ========== TAB 2 ==========
# -----------------------------
with tab2:
    st.header("🧠 Explainable AI (SHAP)")

    st.markdown("""
    This section presents **feature-level explanations** of the machine learning models using SHAP (SHapley Additive Explanations).

    - Isolation Forest: Explained directly using TreeSHAP  
    - Autoencoder: Explained using a surrogate Random Forest model on reconstruction error (computed on a representative sample)
    """)

    st.divider()

    # -------------------------
    # Isolation Forest SHAP
    # -------------------------
    st.subheader("🌲 Isolation Forest — Feature Importance")

    if os.path.exists("reports/shap_if_feature_importance.csv"):
        if_imp = pd.read_csv("reports/shap_if_feature_importance.csv")
        st.markdown("**Top contributing features:**")
        st.dataframe(if_imp.head(10), width='stretch')
    else:
        st.error("Missing: reports/shap_if_feature_importance.csv")

    if os.path.exists("reports/shap_if_summary.png"):
        st.image("reports/shap_if_summary.png", caption="Isolation Forest SHAP Summary Plot", width='stretch')
    else:
        st.error("Missing: reports/shap_if_summary.png")

    st.divider()

    # -------------------------
    # Autoencoder SHAP
    # -------------------------
    st.subheader("🧬 Autoencoder (Surrogate Model) — Feature Importance")

    if os.path.exists("reports/shap_ae_feature_importance.csv"):
        ae_imp = pd.read_csv("reports/shap_ae_feature_importance.csv")
        st.markdown("**Top contributing features:**")
        st.dataframe(ae_imp.head(10), width='stretch')
    else:
        st.error("Missing: reports/shap_ae_feature_importance.csv")

    if os.path.exists("reports/shap_ae_summary.png"):
        st.image("reports/shap_ae_summary.png", caption="Autoencoder SHAP Summary Plot (via Surrogate Model)", width='stretch')
    else:
        st.error("Missing: reports/shap_ae_summary.png")

    st.success("SHAP explanations are precomputed offline and loaded for fast and stable visualization.")
