import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Parameters
# ----------------------------
N_RECORDS = 8000
OUTPUT_PATH = "data/raw/realistic_regulatory_data.csv"

def generate_regulatory_data(n=8000, seed=42):
    np.random.seed(seed)

    # -------------------------------
    # Base reference data
    # -------------------------------
    currencies = ["INR", "USD", "GBP", "EUR", "HKD", "SGD"]
    countries = ["IN", "US", "GB", "DE", "FR", "ES", "IT", "NL", "HK", "SG"]
    counterparties = ["Retail", "Corporate", "Sovereign", "Bank"]
    product_types = ["Loan", "Bond", "Derivative", "Repo"]
    exposure_classes = ["Retail", "Corporate", "Sovereign", "Bank"]

    # -------------------------------
    # Base dataset
    # -------------------------------
    df = pd.DataFrame({
        "Account_ID": np.arange(1, n+1),
        "Report_Date": pd.to_datetime("2025-03-31"),
        "Exposure_Amount": np.random.lognormal(mean=10, sigma=1.0, size=n),
        "Risk_Weight": np.random.uniform(0.1, 1.0, size=n),
        "Country_Code": np.random.choice(countries, size=n),
        "Currency": np.random.choice(currencies, size=n),
        "Counterparty_Type": np.random.choice(counterparties, size=n),
        "Product_Type": np.random.choice(product_types, size=n),
        "Exposure_Class": np.random.choice(exposure_classes, size=n),
        "Credit_Quality_Step": np.random.randint(1, 9, size=n),
        "Maturity_Months": np.random.randint(1, 360, size=n),
        "Is_Collateralized": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    })

    # -------------------------------
    # Derived fields
    # -------------------------------
    df["Capital_Requirement"] = df["Exposure_Amount"] * df["Risk_Weight"] * 0.08
    df["RWA"] = df["Exposure_Amount"] * df["Risk_Weight"]
    df["Exposure_to_Capital_Ratio"] = df["Exposure_Amount"] / (df["Capital_Requirement"] + 1)

    # -------------------------------
    # Inject RULE-BASED anomalies
    # -------------------------------
    df["Injected_Rule_Anomaly"] = 0

    # 1) Missing critical fields
    idx_missing = np.random.choice(df.index, size=int(0.03 * n), replace=False)
    df.loc[idx_missing, "Exposure_Amount"] = np.nan
    df.loc[idx_missing, "Injected_Rule_Anomaly"] = 1

    # 2) Invalid currency codes
    idx_invalid_currency = np.random.choice(df.index, size=int(0.02 * n), replace=False)
    df.loc[idx_invalid_currency, "Currency"] = "XXX"
    df.loc[idx_invalid_currency, "Injected_Rule_Anomaly"] = 1

    # 3) Negative exposures
    idx_negative = np.random.choice(df.index, size=int(0.02 * n), replace=False)
    df.loc[idx_negative, "Exposure_Amount"] = -abs(df.loc[idx_negative, "Exposure_Amount"])
    df.loc[idx_negative, "Injected_Rule_Anomaly"] = 1

    # 4) Risk weight out of bounds
    idx_bad_rw = np.random.choice(df.index, size=int(0.02 * n), replace=False)
    df.loc[idx_bad_rw, "Risk_Weight"] = 1.5
    df.loc[idx_bad_rw, "Injected_Rule_Anomaly"] = 1

    # 5) Capital mismatch
    idx_bad_capital = np.random.choice(df.index, size=int(0.02 * n), replace=False)
    df.loc[idx_bad_capital, "Capital_Requirement"] = df.loc[idx_bad_capital, "Capital_Requirement"] * 0.3
    df.loc[idx_bad_capital, "Injected_Rule_Anomaly"] = 1

    # 6) Duplicates
    dup_idx = np.random.choice(df.index, size=int(0.02 * n), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    df.loc[dup_idx, "Injected_Rule_Anomaly"] = 1

    # -------------------------------
    # Inject ML-ONLY hidden anomalies (IMPORTANT)
    # These DO NOT violate any rule
    # -------------------------------
    df["Injected_ML_Only_Anomaly"] = 0

    clean_idx = df[df["Injected_Rule_Anomaly"] == 0].sample(frac=0.01, random_state=seed).index

    # Strange but valid combinations:
    df.loc[clean_idx, "Exposure_Amount"] = df["Exposure_Amount"].median() * 1.8
    df.loc[clean_idx, "Risk_Weight"] = 0.05   # still valid
    df.loc[clean_idx, "Credit_Quality_Step"] = 1
    df.loc[clean_idx, "Maturity_Months"] = 300
    df.loc[clean_idx, "Is_Collateralized"] = 1
    df.loc[clean_idx, "Product_Type"] = "Derivative"

    # Recalculate derived fields
    df.loc[clean_idx, "Capital_Requirement"] = df.loc[clean_idx, "Exposure_Amount"] * df.loc[clean_idx, "Risk_Weight"] * 0.08
    df.loc[clean_idx, "RWA"] = df.loc[clean_idx, "Exposure_Amount"] * df.loc[clean_idx, "Risk_Weight"]
    df.loc[clean_idx, "Exposure_to_Capital_Ratio"] = df.loc[clean_idx, "Exposure_Amount"] / (df.loc[clean_idx, "Capital_Requirement"] + 1)

    df.loc[clean_idx, "Injected_ML_Only_Anomaly"] = 1

    # -------------------------------
    # Shuffle dataset
    # -------------------------------
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_regulatory_data(n=N_RECORDS)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Generated dataset shape:", df.shape)
    print("Rule anomalies:", df["Injected_Rule_Anomaly"].sum())
    print("ML-only anomalies:", df["Injected_ML_Only_Anomaly"].sum())

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["Exposure_Amount"])
    plt.title("Boxplot of Exposure Amount (Raw Data)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["Risk_Weight"])
    plt.title("Boxplot of Risk Weight (Raw Data)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["Capital_Requirement"])
    plt.title("Boxplot of Capital Requirement (Raw Data)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["RWA"])
    plt.title("Boxplot of Risk Weighted Asset (Raw Data)")
    plt.tight_layout()
    plt.show()

    df["Exposure_to_Capital_Ratio"] = (
    df["Exposure_Amount"] / df["Capital_Requirement"]
    )

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["Exposure_to_Capital_Ratio"])
    plt.title("Boxplot of Exposure-to-Capital Ratio (Raw Data)")
    plt.tight_layout()
    plt.show()

