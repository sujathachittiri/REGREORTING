import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = "data/processed/processed_data.csv"
OUTPUT_PATH = "data/processed/model_scores.csv"

os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_PATH)

# Row identifier for joining later
if "Account_ID" in df.columns:
    ids = df["Account_ID"].values
else:
    ids = np.arange(len(df))

# -----------------------------
# Drop non-ML columns
# -----------------------------
DROP_COLS = [
    "Account_ID",
    "Report_Date",
    "Injected_Rule_Anomaly",
    "Injected_ML_Only_Anomaly",
    "DQ_RULE_SCORE",
    "RULE_ANOMALY_FLAG"
]

X_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# -----------------------------
# Identify feature types
# -----------------------------
categorical_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

# -----------------------------
# Preprocessing (scaling + encoding)
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

X = preprocessor.fit_transform(X_df)

# -----------------------------
# Handle missing values (CRITICAL)
# -----------------------------
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# -----------------------------
# Train models
# -----------------------------

# Isolation Forest
if_model = IsolationForest(contamination=0.05, random_state=42)
if_model.fit(X)
if_scores = -if_model.decision_function(X)

# LOF
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
lof.fit(X)
lof_scores = -lof.decision_function(X)

# OCSVM
ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
ocsvm.fit(X)
ocsvm_scores = -ocsvm.decision_function(X)

# PCA (reconstruction error)
pca = PCA(n_components=0.95, random_state=42)
pca.fit(X)
X_pca_rec = pca.inverse_transform(pca.transform(X))
pca_scores = np.mean((X - X_pca_rec) ** 2, axis=1)

# Autoencoder
input_dim = X.shape[1]
inp = Input(shape=(input_dim,))
x = Dense(64, activation="relu")(inp)
x = Dense(32, activation="relu")(x)
x = Dense(64, activation="relu")(x)
out = Dense(input_dim, activation="linear")(x)

ae = Model(inp, out)
ae.compile(optimizer="adam", loss="mse")

ae.fit(
    X, X,
    validation_split=0.1,
    epochs=50,
    batch_size=256,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=0
)

ae_rec = np.mean(np.square(ae.predict(X, verbose=0)), axis=1)

# -----------------------------
# Combine scores (ensemble)
# -----------------------------
def normalize(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

score_if = normalize(if_scores)
score_lof = normalize(lof_scores)
score_ocsvm = normalize(ocsvm_scores)
score_pca = normalize(pca_scores)
score_ae = normalize(ae_rec)

# Weighted ensemble
ML_SCORE = (
    0.3 * score_if +
    0.1 * score_lof +
    0.1 * score_ocsvm +
    0.2 * score_pca +
    0.3 * score_ae
)

# -----------------------------
# Thresholding
# -----------------------------
threshold = np.quantile(ML_SCORE, 0.95)  # top 5%
ML_ANOMALY_FLAG = (ML_SCORE >= threshold).astype(int)

# -----------------------------
# Save output
# -----------------------------
out_df = pd.DataFrame({
    "ROW_ID": ids,
    "ML_SCORE": ML_SCORE,
    "ML_ANOMALY_FLAG": ML_ANOMALY_FLAG
})

out_df.to_csv(OUTPUT_PATH, index=False)

# -----------------------------
# Summary
# -----------------------------
print("=== MODEL SCORING SUMMARY ===")
print("Rows scored:", len(out_df))
print("ML anomalies flagged:", out_df["ML_ANOMALY_FLAG"].sum())
print("Output written to:", OUTPUT_PATH)
