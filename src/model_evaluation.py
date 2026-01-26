import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# -----------------------------
# Paths
# -----------------------------
FINAL_PATH = "data/processed/final_scored_data.csv"
ML_DETAIL_PATH = "data/processed/model_scores_detailed.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df_final = pd.read_csv(FINAL_PATH)
df_ml = pd.read_csv(ML_DETAIL_PATH)

# -----------------------------
# Ground truth
# -----------------------------
df_final["TRUE_ANOMALY"] = (
    (df_final["Injected_Rule_Anomaly"] == 1) |
    (df_final["Injected_ML_Only_Anomaly"] == 1)
).astype(int)

# -----------------------------
# Helper
# -----------------------------
def precision_at_k(y_true, scores, k=0.05):
    n = int(len(scores) * k)
    idx = np.argsort(scores)[::-1][:n]
    return y_true[idx].mean()

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return np.nan

# -----------------------------
# ========== PART 1: ML MODEL COMPARISON ==========
# -----------------------------
ml_eval = df_ml.merge(df_final[["ROW_ID", "TRUE_ANOMALY"]], on="ROW_ID", how="left")

models = {
    "IsolationForest": "IF_SCORE",
    "LOF": "LOF_SCORE",
    "OCSVM": "OCSVM_SCORE",
    "PCA": "PCA_SCORE",
    "Autoencoder": "AE_SCORE",
    "Ensemble": "ENSEMBLE_SCORE"
}

ml_results = []

for name, col in models.items():
    auc = safe_auc(ml_eval["TRUE_ANOMALY"], ml_eval[col])
    p5 = precision_at_k(ml_eval["TRUE_ANOMALY"].values, ml_eval[col].values, k=0.05)
    ml_results.append({
        "Model": name,
        "ROC_AUC": auc,
        "Precision@5%": p5
    })

ml_results_df = pd.DataFrame(ml_results)
ml_results_df.to_csv(os.path.join(ARTIFACT_DIR, "model_comparison_ml.csv"), index=False)

# -----------------------------
# ========== PART 2: SYSTEM COMPARISON ==========
# -----------------------------
system_results = []

system_results.append({
    "System": "Rule-Based",
    "ROC_AUC": safe_auc(df_final["TRUE_ANOMALY"], df_final["RULE_ANOMALY_FLAG"]),
    "Precision@5%": np.nan
})

system_results.append({
    "System": "ML-Based (Ensemble)",
    "ROC_AUC": safe_auc(df_final["TRUE_ANOMALY"], df_final["ML_SCORE"]),
    "Precision@5%": precision_at_k(df_final["TRUE_ANOMALY"].values, df_final["ML_SCORE"].values, k=0.05)
})

system_results.append({
    "System": "Hybrid (Rule + ML)",
    "ROC_AUC": safe_auc(df_final["TRUE_ANOMALY"], df_final["FINAL_ANOMALY_FLAG"]),
    "Precision@5%": np.nan
})

system_df = pd.DataFrame(system_results)
system_df.to_csv(os.path.join(ARTIFACT_DIR, "model_comparison_system.csv"), index=False)

# -----------------------------
# ========== PART 3: OVERLAP ANALYSIS ==========
# -----------------------------
rule_only = ((df_final["RULE_ANOMALY_FLAG"] == 1) & (df_final["ML_ANOMALY_FLAG"] == 0)).sum()
ml_only = ((df_final["RULE_ANOMALY_FLAG"] == 0) & (df_final["ML_ANOMALY_FLAG"] == 1)).sum()
both = ((df_final["RULE_ANOMALY_FLAG"] == 1) & (df_final["ML_ANOMALY_FLAG"] == 1)).sum()

with open(os.path.join(ARTIFACT_DIR, "overlap_analysis.txt"), "w") as f:
    f.write("=== OVERLAP ANALYSIS ===\n")
    f.write(f"Total rows: {len(df_final)}\n")
    f.write(f"True anomalies: {df_final['TRUE_ANOMALY'].sum()}\n")
    f.write(f"Rule-only anomalies: {rule_only}\n")
    f.write(f"ML-only anomalies: {ml_only}\n")
    f.write(f"Detected by both: {both}\n")
    f.write(f"Final anomalies: {df_final['FINAL_ANOMALY_FLAG'].sum()}\n")

# -----------------------------
# Print summary
# -----------------------------
print("=== ML MODEL COMPARISON ===")
print(ml_results_df)
print()

print("=== SYSTEM COMPARISON ===")
print(system_df)
print()

print("=== OVERLAP ===")
print("Rule-only:", rule_only)
print("ML-only:", ml_only)
print("Both:", both)

print()
print("Artifacts written to:", ARTIFACT_DIR)
