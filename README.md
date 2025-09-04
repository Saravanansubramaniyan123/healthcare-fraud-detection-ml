# Healthcare Provider Fraud Detection (Medicare Claims)

End-to-end machine-learning pipeline to flag **potentially fraudulent healthcare providers** from Medicare claim patterns. The repository is **code-first (Python)** and all **CSVs in this repo are program-generated outputs** (processed datasets and submissions) — not the original raw Kaggle files.

---

## 🩺 Problem Statement
Healthcare fraud is a major issue in the insurance industry, where providers submit false or misleading claims to obtain unwarranted reimbursements. This leads to billions of dollars in financial losses and impacts patients who genuinely need care.

The task is to analyze Medicare claims data (beneficiary, inpatient, outpatient) and build a model to predict whether a provider is **potentially fraudulent**.

---

## 🎯 Objective
- Merge multiple raw claim tables into a **provider-level dataset**.
- Engineer meaningful features that capture provider behavior.
- Train machine learning models to predict the target variable `PotentialFraud` (Yes/No).
- Compare models and choose one that balances **accuracy, recall, and interpretability**.
- Save outputs (models, metadata, predictions) for reproducibility.

---

## ⚡ Key Challenges
- **Imbalanced Data:** Only a small fraction of providers are fraudulent.
- **Heterogeneous Sources:** Data comes from inpatient, outpatient, and beneficiary tables.
- **Feature Engineering:** Must aggregate claim-level data into provider-level patterns.
- **Business Relevance:** Prioritize recall (catching fraud) without too many false alarms.

---

## 💡 Solution Overview
1. **Data Processing:** Cleaned and merged multiple claim tables into provider-level records.
2. **Feature Engineering:** Built features on claim counts, reimbursement patterns, length of stay, chronic conditions, and procedure diversity.
3. **Modeling:** Trained Logistic Regression, Random Forest, and Isolation Forest models.
4. **Evaluation:** Used metrics like Accuracy, ROC-AUC, Precision, Recall, and F1-score.
5. **Deployment Artifacts:** Saved models (`.joblib`), metadata (`.json`), and outputs (`.csv`) for reproducibility.

---

## 📊 Dataset
- **Source:** Kaggle — [Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data)
- **Kaggle link (provided):** https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data
- **Tables:** Beneficiary, Inpatient, Outpatient, and Provider label table.
- **Target:** `PotentialFraud` (Yes/No) at provider level.

> 🔎 **Note:** Raw Kaggle CSVs are **not** committed. The repository contains **processed CSVs** and **submission CSVs** that are **saved automatically by the program** during runs.

---

## 🗂️ Repository Structure
```
├─ data/                          # All program-generated CSVs
│  ├─ processed/                  # Engineered datasets at provider level
│  └─ submissions/                # Final prediction CSVs
│
├─ models/                        # Saved models + transformers + metadata
│  ├─ logistic_regression_threshold_60.joblib
│  ├─ random_forest.joblib
│  ├─ isolation_forest.joblib
│  ├─ isolation_forest_pca_scaler.joblib
│  ├─ isolation_forest_pca_transformer.joblib
│  ├─ logistic_regression_scaler.joblib
│  ├─ *metadata.json
│  └─ ...
│
├─ src/                           # Source code
│  └─ healthcare_fraud_detection.py
│
├─ README.md
└─ requirements.txt (to be added)
```

---

## 📈 Model Performance
The models were evaluated on a validation dataset using multiple metrics.

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|:--------:|:---------:|:------:|:--------:|:------:|
| Logistic Regression  | 92.3%    | 57.5%     | 68.4%  | 62.4%    | 81.6%  |
| Random Forest        | 87.2%    | 42.0%     | 86.8%  | 55.9%    | 87.0%  |
| Isolation Forest     | 92.0%    | 55.0%     | 53.0%  | 54.0%    |   –    |

- **Logistic Regression** chosen for balanced performance and interpretability.
- **Random Forest** provides higher recall, useful if the goal is to maximize fraud detection.
- **Isolation Forest** used as an unsupervised anomaly detection baseline.

---

## ⚙️ Setup
```bash
# 1) Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/Mac

# 2) Install dependencies
pip install -r requirements.txt
```
**Suggested requirements (edit if versions differ):**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.11
matplotlib>=3.7
joblib>=1.3
```

---

## 🧪 How to Run
### Run full pipeline
```bash
python src/healthcare_fraud_detection.py
```
This will:
- Load raw/processed data (depending on your setup)
- Engineer features
- Train models (LR, RF, IF)
- Apply thresholds (e.g., 0.60 for Logistic Regression)
- Save artifacts → `models/`
- Save CSV outputs → `data/processed/` and `data/submissions/`

---

## 🔮 Inference Example
```python
import joblib, pandas as pd

X = pd.read_csv("data/processed/provider_features.csv")
clf = joblib.load("models/logistic_regression_threshold_60.joblib")
proba = clf.predict_proba(X)[:, 1]
X.assign(fraud_risk=proba).to_csv("data/submissions/predictions_lr.csv", index=False)
```

---

## 🧭 Responsible Use
This project flags **potential** fraud for review. It does **not** prove actual fraud. Always validate with domain experts and policies.

---

## 🙌 Acknowledgments
- Kaggle dataset: *Healthcare Provider Fraud Detection Analysis*.
- Python ecosystem: pandas, scikit-learn, numpy, imbalanced-learn, matplotlib.


