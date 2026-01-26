import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# Paths
# -----------------------------
FINAL_PATH = "data/processed/final_scored_data.csv"
ML_PATH = "data/processed/model_scores_detailed.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(FINAL_PATH)
scores = pd.read_csv(ML_PATH)

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
X = X.select_dtypes(include=[np.number])

print("Using features for AE SHAP:", X.columns.tolist())

# -----------------------------
# Target = Autoencoder score
# -----------------------------
y = scores["AE_SCORE"].values

# Align lengths
min_len = min(len(X), len(y))
X = X.iloc[:min_len]
y = y[:min_len]

# -----------------------------
# Scale
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# SUBSAMPLE FOR SPEED
# -----------------------------
MAX_SHAP_ROWS = 600

if X_scaled.shape[0] > MAX_SHAP_ROWS:
    rng = np.random.RandomState(42)
    idx = rng.choice(X_scaled.shape[0], MAX_SHAP_ROWS, replace=False)
    X_shap = X_scaled[idx]
    y_shap = y[idx]
    X_shap_df = X.iloc[idx]
else:
    X_shap = X_scaled
    y_shap = y
    X_shap_df = X.copy()

print("Rows used for AE SHAP:", X_shap.shape[0])

# -----------------------------
# Train surrogate model
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_shap, y_shap)

# -----------------------------
# SHAP
# -----------------------------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_shap)

# -----------------------------
# Global importance
# -----------------------------
mean_abs = np.abs(shap_values).mean(axis=0)

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsSHAP": mean_abs
}).sort_values("MeanAbsSHAP", ascending=False)

# Save CSV
imp_df.to_csv(os.path.join(ARTIFACT_DIR, "shap_ae_feature_importance.csv"), index=False)

# Plot
plt.figure()
shap.summary_plot(shap_values, X_shap_df, show=False)
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "shap_ae_summary.png"), dpi=150)

print("=== AE SURROGATE SHAP DONE ===")
print("Saved:")
print(" - artifacts/shap_ae_feature_importance.csv")
print(" - artifacts/shap_ae_summary.png")
