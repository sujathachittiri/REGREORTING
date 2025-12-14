"""
Model Selection Rationale:
- Autoencoder showed highest ROC-AUC and captures non-linear anomalies
- Isolation Forest selected for scalability and explainability (SHAP)
- LOF excluded due to batch scalability limitations
- OCSVM excluded due to hyperparameter sensitivity

Final approach uses hybrid IF + AE scoring.
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import os

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/processed/dq_report.csv")

# Create weak labels using DQ rules (for evaluation only)
df["Weak_Label"] = (df["DQ_RULE_SCORE"] > 0).astype(int)

features = [
    "Exposure_Amount",
    "Risk_Weight",
    "Capital_Requirement",
    "Tenor_Months"
]

df_model = df[features].fillna(0)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df_model)

# -----------------------------
# Isolation Forest
# -----------------------------
if_model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
if_scores = -if_model.fit_predict(X)

# -----------------------------
# Local Outlier Factor
# -----------------------------
lof_model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)
lof_scores = -lof_model.fit_predict(X)

# -----------------------------
# One-Class SVM
# -----------------------------
ocsvm = OneClassSVM(
    kernel="rbf",
    nu=0.05,
    gamma="scale"
)
ocsvm.fit(X)
ocsvm_scores = -ocsvm.predict(X)

# -----------------------------
# Autoencoder
# -----------------------------
input_dim = X.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(
    X, X,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

reconstructions = autoencoder.predict(X)
ae_scores = np.mean(np.square(X - reconstructions), axis=1)

# -----------------------------
# Evaluation
# -----------------------------
results = []

for name, scores in {
    "IsolationForest": if_scores,
    "LOF": lof_scores,
    "OCSVM": ocsvm_scores,
    "Autoencoder": ae_scores
}.items():
    try:
        auc = roc_auc_score(df["Weak_Label"], scores)
    except:
        auc = np.nan
    results.append({
        "Model": name,
        "ROC_AUC": auc,
        "Avg_Score": np.mean(scores)
    })

summary_df = pd.DataFrame(results)

# -----------------------------
# Save outputs
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

df_scores = df.copy()
df_scores["IF_Score"] = if_scores
df_scores["LOF_Score"] = lof_scores
df_scores["OCSVM_Score"] = ocsvm_scores
df_scores["AE_Score"] = ae_scores

df_scores.to_csv("data/processed/model_scores.csv", index=False)
summary_df.to_csv("data/processed/model_comparison_summary.csv", index=False)

print("Model comparison completed")
print(summary_df)