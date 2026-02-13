import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

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

def plot_roc(y_true, scores, label):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc_val = auc(fpr, tpr) # Renamed to avoid shadowing
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_val:.2f})")

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


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
    current_auc = safe_auc(ml_eval["TRUE_ANOMALY"], ml_eval[col]) # Renamed local variable
    p5 = precision_at_k(ml_eval["TRUE_ANOMALY"].values, ml_eval[col].values, k=0.05)
    ml_results.append({
        "Model": name,
        "ROC_AUC": current_auc,
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

print("=== ROC Curves for Individual ML Models ===")
plt.figure(figsize=(6,5))

plot_roc(df_final["TRUE_ANOMALY"], df_final["IF_SCORE"], "Isolation Forest")
plot_roc(df_final["TRUE_ANOMALY"], df_final["AE_SCORE"], "Autoencoder")
plot_roc(df_final["TRUE_ANOMALY"], df_final["LOF_SCORE"], "Local Outlier Factor")
plot_roc(df_final["TRUE_ANOMALY"], df_final["OCSVM_SCORE"], "One Class SVM")
plot_roc(df_final["TRUE_ANOMALY"], df_final["PCA_SCORE"], "PCA")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Individual ML Models")
plt.legend()
plt.tight_layout()
plt.show()

print("=== ROC Curves for Ensemble Model ===")
plt.figure(figsize=(6,5))

plot_roc(df_final["TRUE_ANOMALY"], df_final["ENSEMBLE_SCORE"], "ML Ensemble")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – ML Ensemble")
plt.legend()
plt.tight_layout()
plt.show()

print("=== SYSTEM COMPARISON ===")
print(system_df)
print()

print("=== OVERLAP ===")
print("Rule-only:", rule_only)
print("ML-only:", ml_only)
print("Both:", both)

print("=== ROC Curve for Hybrid Detection System ===")

hybrid_score = df_final["FINAL_ANOMALY_FLAG"]

plt.figure(figsize=(6,5))
fpr, tpr, _ = roc_curve(df_final["TRUE_ANOMALY"], hybrid_score)
fpr, tpr, _ = roc_curve(df_final["TRUE_ANOMALY"], df_final["RULE_ANOMALY_FLAG"])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"Hybrid System (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Hybrid System")
plt.legend()
plt.tight_layout()
plt.show()

print("=== ROC Curves for SYSTEM MODELS ===")
plt.figure(figsize=(6,5))

plot_roc(df_final["TRUE_ANOMALY"], df_final["RULE_ANOMALY_FLAG"], "Rule Based System")
plot_roc(df_final["TRUE_ANOMALY"], df_final["ML_ANOMALY_FLAG"], "ML Based System")
plot_roc(df_final["TRUE_ANOMALY"], df_final["FINAL_ANOMALY_FLAG"], "Hybrid System")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – System Models")
plt.legend()
plt.tight_layout()
plt.show()

print("=== CONFUSION MATRIX FOR SYSTEM MODELS ===")

plot_conf_matrix(
    df_final["TRUE_ANOMALY"],
    df_final["RULE_ANOMALY_FLAG"],
    "Confusion Matrix – Rule-Based Detection"
)

plot_conf_matrix(
    df_final["TRUE_ANOMALY"],
    df_final["ML_ANOMALY_FLAG"],
    "Confusion Matrix – ML-Based Detection"
)

plot_conf_matrix(
    df_final["TRUE_ANOMALY"],
    df_final["FINAL_ANOMALY_FLAG"],
    "Confusion Matrix – Hybrid (Rule + ML) Detection"
)


print()
print("Artifacts written to:", ARTIFACT_DIR)