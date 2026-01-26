import os
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
RULE_PATH = "data/processed/processed_data.csv"
ML_PATH = "data/processed/model_scores.csv"
OUTPUT_PATH = "data/processed/final_scored_data.csv"

os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df_rules = pd.read_csv(RULE_PATH)
df_ml = pd.read_csv(ML_PATH)

# -----------------------------
# Create join key
# -----------------------------
# We assume ROW_ID in ML corresponds to Account_ID or row index

if "Account_ID" in df_rules.columns:
    df_rules["ROW_ID"] = df_rules["Account_ID"]
else:
    df_rules["ROW_ID"] = df_rules.index

# -----------------------------
# Merge
# -----------------------------
df = df_rules.merge(df_ml, on="ROW_ID", how="left")

# Safety fill (in case some rows missing)
df["ML_SCORE"] = df["ML_SCORE"].fillna(0.0)
df["ML_ANOMALY_FLAG"] = df["ML_ANOMALY_FLAG"].fillna(0).astype(int)

# -----------------------------
# Final anomaly decision
# -----------------------------
df["FINAL_ANOMALY_FLAG"] = (
    (df["RULE_ANOMALY_FLAG"] == 1) | (df["ML_ANOMALY_FLAG"] == 1)
).astype(int)

# -----------------------------
# Save
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)

# -----------------------------
# Summary
# -----------------------------
print("=== UNIFIED SCORING SUMMARY ===")
print("Rows:", len(df))
print("Rule anomalies:", df["RULE_ANOMALY_FLAG"].sum())
print("ML anomalies:", df["ML_ANOMALY_FLAG"].sum())
print("Final anomalies:", df["FINAL_ANOMALY_FLAG"].sum())
print("ML-only anomalies:", ((df["RULE_ANOMALY_FLAG"] == 0) & (df["ML_ANOMALY_FLAG"] == 1)).sum())
print("Output written to:", OUTPUT_PATH)
