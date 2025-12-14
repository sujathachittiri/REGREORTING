import pandas as pd
import json
import os
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load config
# -----------------------------
with open("dq_config.json", "r") as f:
    CONFIG = json.load(f)

# -----------------------------
# Load inputs
# -----------------------------
dq_df = pd.read_csv("data/processed/dq_report.csv")
ml_df = pd.read_csv("data/processed/model_scores.csv")

# -----------------------------
# Merge DQ + ML outputs
# -----------------------------
df = dq_df.merge(
    ml_df[
        [
            "Account_ID",
            "IF_Score",
            "AE_Score"
        ]
    ],
    on="Account_ID",
    how="left"
)

# -----------------------------
# Normalize ML scores
# -----------------------------
scaler = MinMaxScaler()
df[["IF_Score_Norm", "AE_Score_Norm"]] = scaler.fit_transform(
    df[["IF_Score", "AE_Score"]]
)

# -----------------------------
# Hybrid ML anomaly score
# -----------------------------
if_weight = CONFIG["ml_thresholds"]["isolation_forest_weight"]
ae_weight = CONFIG["ml_thresholds"]["autoencoder_weight"]

df["ML_Anomaly_Score"] = (
    if_weight * df["IF_Score_Norm"] +
    ae_weight * df["AE_Score_Norm"]
)

# -----------------------------
# Final anomaly decision
# -----------------------------
ml_threshold = CONFIG["ml_thresholds"]["final_ml_anomaly_threshold"]

df["ML_Anomaly_Flag"] = (df["ML_Anomaly_Score"] > ml_threshold).astype(int)

df["FINAL_ANOMALY_FLAG"] = (
    (df["DQ_RULE_ANOMALY"] == 1) |
    (df["ML_Anomaly_Flag"] == 1)
).astype(int)

# -----------------------------
# Severity score (optional but strong)
# -----------------------------
df["ANOMALY_SEVERITY"] = (
    df["DQ_RULE_SCORE"] +
    (df["ML_Anomaly_Score"] * 10)
).round(2)

# -----------------------------
# Save output
# -----------------------------
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/merged_dq_ml_report.csv", index=False)

print("Unified anomaly scoring completed")
print("Output saved to data/processed/merged_dq_ml_report.csv")
print(df[["FINAL_ANOMALY_FLAG"]].value_counts())
