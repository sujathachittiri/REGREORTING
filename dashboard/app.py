import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Intelligent Data Quality Monitoring",
    layout="wide"
)

st.title("Intelligent Data Quality Monitoring for Regulatory Reporting")
st.caption("Hybrid Rule-based + ML-driven Anomaly Detection (IF + Autoencoder)")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/merged_dq_ml_report.csv")

df = load_data()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

country = st.sidebar.multiselect(
    "Country",
    options=df["Country_Code"].unique(),
    default=df["Country_Code"].unique()
)

final_flag = st.sidebar.selectbox(
    "Final Anomaly Flag",
    options=["All", 0, 1]
)

filtered_df = df[df["Country_Code"].isin(country)]

if final_flag != "All":
    filtered_df = filtered_df[filtered_df["FINAL_ANOMALY_FLAG"] == final_flag]

# -----------------------------
# KPI Section
# -----------------------------
total_records = len(filtered_df)
total_anomalies = filtered_df["FINAL_ANOMALY_FLAG"].sum()
dq_anomalies = filtered_df["DQ_RULE_ANOMALY"].sum()
ml_anomalies = filtered_df["ML_Anomaly_Flag"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", total_records)
col2.metric("Final Anomalies", int(total_anomalies))
col3.metric("DQ Rule Violations", int(dq_anomalies))
col4.metric("ML-only Anomalies", int(ml_anomalies))

st.divider()

# -----------------------------
# Anomaly Trend
# -----------------------------
st.subheader("Anomaly Trend Over Time")

trend_df = (
    filtered_df
    .groupby("Report_Date")["FINAL_ANOMALY_FLAG"]
    .sum()
    .reset_index()
)

st.line_chart(trend_df.set_index("Report_Date"))

# -----------------------------
# Severity Distribution
# -----------------------------
st.subheader("Anomaly Severity Distribution")

st.bar_chart(
    filtered_df["ANOMALY_SEVERITY"].value_counts().sort_index()
)

# -----------------------------
# Detailed Table
# -----------------------------
st.subheader("Anomalous Records (Drill-down View)")

columns_to_show = [
    "Account_ID",
    "Report_Date",
    "Country_Code",
    "Exposure_Amount",
    "Risk_Weight",
    "DQ_RULE_SCORE",
    "ML_Anomaly_Score",
    "ANOMALY_SEVERITY",
    "FINAL_ANOMALY_FLAG"
]

st.dataframe(
    filtered_df[columns_to_show]
    .sort_values("ANOMALY_SEVERITY", ascending=False),
    #use_container_width=True
    width='stretch'
)

# -----------------------------
# Explainability Section
# -----------------------------
st.subheader("Explainability (Conceptual)")

st.info(
    """
    This dashboard provides transparent and auditable explanations for all detected anomalies.

    Rule-based explainability:
    Deterministic data quality rules identify explicit regulatory violations such as missing
    mandatory fields, invalid reference data, negative exposure values, duplicates, and
    threshold breaches. These violations are reflected in the rule-based score.

    Machine learning explainability:
    ML anomaly scores quantify how much a record deviates from learned historical data
    patterns, highlighting unusual or unexpected behaviour.

    Decision transparency:
    A record is flagged as anomalous when it violates at least one data quality rule or when
    its ML anomaly score exceeds the defined threshold.
    """
)

st.caption(
    "Records with higher anomaly severity require higher priority investigation based on combined rule violations and ML anomaly scores."
)

st.success("Dashboard loaded successfully")
