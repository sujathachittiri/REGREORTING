import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ----------------------------
# Parameters
# ----------------------------
N_RECORDS = 6000
OUTPUT_PATH = "data/raw/realistic_regulatory_data.csv"

countries = ["IN", "US", "GB", "DE", "FR", "ES", "IT", "NL", "HK", "SG"]
currency_map = {
    "IN": "INR", "US": "USD", "GB": "GBP", "DE": "EUR",
    "FR": "EUR", "ES": "EUR", "IT": "EUR",
    "NL": "EUR", "HK": "HKD", "SG": "SGD"
}

counterparty_types = ["Retail", "Corporate", "SME", "Bank"]
product_types = ["Loan", "Mortgage", "CreditCard", "Overdraft"]
source_systems = ["CoreBanking", "Treasury", "RiskEngine"]

# ----------------------------
# Generate base data
# ----------------------------
data = []

for i in range(N_RECORDS):
    country = np.random.choice(countries)
    exposure = np.random.lognormal(mean=10, sigma=1.0)
    risk_weight = np.clip(np.random.normal(0.5, 0.2), 0, 1)
    capital = exposure * risk_weight * 0.08

    record = {
        "Account_ID": f"ACC_{100000 + i}",
        "Report_Date": (datetime.today() - timedelta(days=np.random.randint(0, 90))).date(),
        "Exposure_Amount": round(exposure, 2),
        "Risk_Weight": round(risk_weight, 2),
        "Capital_Requirement": round(capital, 2),
        "Country_Code": country,
        "Currency": currency_map[country],
        "Counterparty_Type": np.random.choice(counterparty_types),
        "Product_Type": np.random.choice(product_types),
        "Source_System": np.random.choice(source_systems),
        "Tenor_Months": np.random.randint(6, 360)
    }
    data.append(record)

df = pd.DataFrame(data)

# ----------------------------
# Inject data quality issues
# ----------------------------

# 1. Missing values
for col in ["Exposure_Amount", "Risk_Weight", "Country_Code"]:
    df.loc[df.sample(frac=0.02).index, col] = None

# 2. Invalid currencies
df.loc[df.sample(frac=0.01).index, "Currency"] = "XXX"

# 3. Negative exposure
df.loc[df.sample(frac=0.01).index, "Exposure_Amount"] *= -1

# 4. Risk weight out of bounds
df.loc[df.sample(frac=0.01).index, "Risk_Weight"] = 1.5

# 5. Duplicate records
dup_rows = df.sample(frac=0.01)
df = pd.concat([df, dup_rows], ignore_index=True)

# ----------------------------
# Save dataset
# ----------------------------
os.makedirs("data/raw", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Synthetic regulatory dataset generated successfully")
print(f"Records: {len(df)}")
print(f"Saved to: {OUTPUT_PATH}")
