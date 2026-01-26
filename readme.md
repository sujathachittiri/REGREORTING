\# Intelligent Data Quality Monitoring for Regulatory Reporting



This project implements a hybrid data quality monitoring framework for

financial regulatory reporting systems (PRA / ECB).



Pipeline:

1\. Rule-based Data Quality Engine

2\. Machine Learning anomaly detection (Isolation Forest + Autoencoder)

3\. Unified anomaly scoring

4\. Dashboard visualisation



Technology:

\- Python, Scikit-learn

\- Autoencoders (Deep Learning)

\- SHAP explainability

\- Streamlit dashboard



This project is part of the M.Tech AIML dissertation at BITS Pilani (WILP).



\## Model Selection and Justification



Multiple unsupervised anomaly detection algorithms were evaluated to identify

data quality issues in regulatory reporting data. The models compared include

Isolation Forest (IF), Local Outlier Factor (LOF), One-Class SVM (OCSVM), and

Autoencoders (AE).



\### Evaluation Summary

| Model | ROC-AUC | Observations |

|------|--------|--------------|

| Isolation Forest | ~0.58 | Scalable, robust to high-dimensional data, supports explainability |

| LOF | ~0.63 | Slightly better ROC-AUC but not suitable for large-scale batch processing |

| One-Class SVM | ~0.61 | Sensitive to hyperparameters and computationally expensive |

| Autoencoder | ~0.69 | Best ROC-AUC, captures non-linear relationships |



\### Final Model Choice: Isolation Forest + Autoencoder



Although LOF and Autoencoder achieved higher ROC-AUC values, LOF was not selected

due to scalability limitations and lack of model explainability. Autoencoders,

while powerful in capturing non-linear patterns, lack direct interpretability.



Isolation Forest was selected due to its scalability, robustness, and compatibility

with explainable AI techniques such as SHAP. A hybrid approach combining Isolation

Forest and Autoencoder was therefore adopted to balance interpretability, scalability,

and detection performance.



This hybrid design aligns with regulatory expectations for explainability,

auditability, and operational robustness.

