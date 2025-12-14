import pandas as pd
import numpy as np
import json

# -----------------------------
# Load configuration
# -----------------------------
with open("dq_config.json", "r") as f:
    CONFIG = json.load(f)

# -----------------------------
# Load raw data
# -----------------------------
df = pd.read_csv("data/raw/realistic_regulatory_data.csv")

# -----------------------------
# Data Quality Rules
# -----------------------------

# 1. Missing mandatory fields
mandatory_cols = CONFIG["critical_fields"]
df["DQ_MISSING_FLAG"] = df[mandatory_cols].isnull().any(axis=1).astype(int)

# 2. Invalid currency
allowed_ccy = CONFIG["code_lists"]["allowed_currencies"]
df["DQ_INVALID_CURRENCY_FLAG"] = (~df["Currency"].isin(allowed_ccy)).astype(int)

# 3. Invalid country
allowed_country = CONFIG["code_lists"]["iso_country_subset"]
df["DQ_INVALID_COUNTRY_FLAG"] = (~df["Country_Code"].isin(allowed_country)).astype(int)

# 4. Negative exposure
df["DQ_NEGATIVE_EXPOSURE_FLAG"] = (df["Exposure_Amount"] < 0).astype(int)

# 5. Risk weight bounds
rw_min = CONFIG["bounds"]["risk_weight_min"]
rw_max = CONFIG["bounds"]["risk_weight_max"]
df["DQ_RISK_WEIGHT_FLAG"] = (
    (df["Risk_Weight"] < rw_min) | (df["Risk_Weight"] > rw_max)
).astype(int)

# 6. Duplicate records
dup_keys = CONFIG["duplication"]["duplicate_key"]
df["DQ_DUPLICATE_FLAG"] = df.duplicated(subset=dup_keys, keep=False).astype(int)

# 7. Exposure outliers (country median based)
median_exp = df.groupby("Country_Code")["Exposure_Amount"].transform("median")
df["DQ_OUTLIER_FLAG"] = (
    df["Exposure_Amount"] > CONFIG["outlier_detection"]["multiple"] * median_exp
).astype(int)

# -----------------------------
# Final Rule Score
# -----------------------------
dq_flags = [col for col in df.columns if col.startswith("DQ_")]
df["DQ_RULE_SCORE"] = df[dq_flags].sum(axis=1)
df["DQ_RULE_ANOMALY"] = (df["DQ_RULE_SCORE"] > 0).astype(int)

# -----------------------------
# Save output
# -----------------------------
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/dq_report.csv", index=False)

print("DQ Engine executed successfully")
print("Output saved to data/processed/dq_report.csv")