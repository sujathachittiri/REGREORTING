import os
import json
import pandas as pd
import numpy as np

# -----------------------------
# Config loading
# -----------------------------
CONFIG_PATH = "dq_config.json"
RAW_DATA_PATH = "data/raw/realistic_regulatory_data.csv"
OUTPUT_PATH = "data/processed/processed_data.csv"

os.makedirs("data/processed", exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(RAW_DATA_PATH)

# -----------------------------
# Initialize rule results
# -----------------------------
df["DQ_RULE_SCORE"] = 0
df["RULE_ANOMALY_FLAG"] = 0

# -----------------------------
# Helper to add rule violation
# -----------------------------
def flag_violation(mask):
    df.loc[mask, "DQ_RULE_SCORE"] += 1

# -----------------------------
# RULE 1: Missing critical fields
# -----------------------------
critical_fields = config.get("critical_fields", [])
for col in critical_fields:
    if col in df.columns:
        flag_violation(df[col].isna())

# -----------------------------
# RULE 2: Invalid currency codes
# -----------------------------
allowed_currencies = config["code_lists"]["allowed_currencies"]
if "Currency" in df.columns:
    flag_violation(~df["Currency"].isin(allowed_currencies))

# -----------------------------
# RULE 3: Invalid country codes
# -----------------------------
allowed_countries = config["code_lists"]["iso_country_subset"]
if "Country_Code" in df.columns:
    flag_violation(~df["Country_Code"].isin(allowed_countries))

# -----------------------------
# RULE 4: Exposure must be >= 0
# -----------------------------
if "Exposure_Amount" in df.columns:
    flag_violation(df["Exposure_Amount"] < 0)

# -----------------------------
# RULE 5: Risk weight bounds
# -----------------------------
rw_min = config["bounds"]["risk_weight_min"]
rw_max = config["bounds"]["risk_weight_max"]
if "Risk_Weight" in df.columns:
    flag_violation((df["Risk_Weight"] < rw_min) | (df["Risk_Weight"] > rw_max))

# -----------------------------
# RULE 6: Tenor bounds (if present)
# -----------------------------
if "Maturity_Months" in df.columns:
    tmin = config["bounds"]["tenor_min_months"]
    tmax = config["bounds"]["tenor_max_months"]
    flag_violation((df["Maturity_Months"] < tmin) | (df["Maturity_Months"] > tmax))

# -----------------------------
# RULE 7: Capital formula consistency
# -----------------------------
if all(col in df.columns for col in ["Exposure_Amount", "Risk_Weight", "Capital_Requirement"]):
    expected_capital = df["Exposure_Amount"] * df["Risk_Weight"] * config["capital_check"]["capital_formula_multiplier"]
    tolerance = config["capital_check"]["capital_tolerance_percentage"]
    diff_ratio = (df["Capital_Requirement"] - expected_capital).abs() / (expected_capital + 1e-6)
    flag_violation(diff_ratio > tolerance)

# -----------------------------
# RULE 8: Duplicate keys
# -----------------------------
dup_keys = config["duplication"]["duplicate_key"]
existing_dup_keys = [k for k in dup_keys if k in df.columns]
if len(existing_dup_keys) > 0:
    dup_mask = df.duplicated(subset=existing_dup_keys, keep=False)
    flag_violation(dup_mask)

# -----------------------------
# Final rule anomaly flag
# -----------------------------
df["RULE_ANOMALY_FLAG"] = (df["DQ_RULE_SCORE"] > 0).astype(int)

# -----------------------------
# Save output
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)

# -----------------------------
# Summary print
# -----------------------------
print("=== DQ ENGINE SUMMARY ===")
print("Input rows:", len(df))
print("Rows with rule violations:", df["RULE_ANOMALY_FLAG"].sum())
print("Average rule violations per row:", df["DQ_RULE_SCORE"].mean())
print("Output written to:", OUTPUT_PATH)
