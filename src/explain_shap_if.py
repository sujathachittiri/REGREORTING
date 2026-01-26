import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/processed/final_scored_data.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Clean feature selection
# -----------------------------
DROP_COLS = [
    # Identifiers
    "Account_ID", "ROW_ID", "Report_Date",

    # Labels / flags
    "Injected_Rule_Anomaly", "Injected_ML_Only_Anomaly",
    "DQ_RULE_SCORE", "RULE_ANOMALY_FLAG",
    "ML_ANOMALY_FLAG", "FINAL_ANOMALY_FLAG",

    # Model outputs
    "IF_SCORE", "LOF_SCORE", "OCSVM_SCORE", "PCA_SCORE", "AE_SCORE",
    "ENSEMBLE_SCORE", "ML_SCORE"
]

features = [c for c in df.columns if c not in DROP_COLS]

X = df[features].copy()

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

print("Using features for IF SHAP:", X.columns.tolist())

# -----------------------------
# Scale
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train Isolation Forest
# -----------------------------
if_model = IsolationForest(contamination=0.05, random_state=42)
if_model.fit(X_scaled)

# -----------------------------
# SHAP
# -----------------------------
explainer = shap.TreeExplainer(if_model)
shap_values = explainer.shap_values(X_scaled)

# -----------------------------
# Global importance
# -----------------------------
mean_abs = np.abs(shap_values).mean(axis=0)

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsSHAP": mean_abs
}).sort_values("MeanAbsSHAP", ascending=False)

# Save CSV
imp_df.to_csv(os.path.join(ARTIFACT_DIR, "shap_if_feature_importance.csv"), index=False)

# Plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "shap_if_summary.png"), dpi=150)

print("=== IF SHAP DONE ===")
print("Saved:")
print(" - artifacts/shap_if_feature_importance.csv")
print(" - artifacts/shap_if_summary.png")
