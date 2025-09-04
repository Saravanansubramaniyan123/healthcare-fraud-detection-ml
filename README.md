# Healthcare Provider Fraud Detection (Medicare Claims)

End‑to‑end machine‑learning pipeline to flag **potentially fraudulent healthcare providers** from Medicare claim patterns. The repository is **code‑first (Python)** and all **CSVs in this repo are program‑generated outputs** (processed datasets and submissions) — not the original raw Kaggle files.

---

## 🚀 TL;DR
- **Goal:** Predict provider‑level fraud (`PotentialFraud`) using claims and beneficiary data.
- **Approach:** Clean & merge → feature engineering → imbalance handling → model training (LR, RF, IF) → threshold tuning.
- **Repo layout:** Professional folder structure (`data/`, `models/`, `src/`, `README.md`).

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

> ✅ This structure is clean and professional: `data/` for outputs, `models/` for joblibs/jsons, `src/` for main Python code, and `README.md` at the top level.

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

### (Optional) Run with explicit args
```bash
python src/healthcare_fraud_detection.py \
  --raw_dir data/raw \
  --processed_dir data/processed \
  --models_dir models \
  --submissions_dir data/submissions \
  --model lr --threshold 0.60 --seed 42
```

---

## 📦 Outputs (Program‑Saved)
- **Models & Transformers** in `models/`
- **Processed CSVs** in `data/processed/`
- **Submissions** in `data/submissions/`

These are committed to repo for easy reproducibility.

---

## 🏗️ Methodology
1. **Merge raw tables** → provider level.
2. **Engineer features** → claim counts, reimbursements, LOS, beneficiary diversity.
3. **Handle imbalance** → class weights / resampling.
4. **Train models** → Logistic Regression, Random Forest, Isolation Forest.
5. **Evaluate** → Accuracy, Precision, Recall, F1, ROC‑AUC.
6. **Persist** → Models + scalers + metadata JSON + CSVs.

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
- Python ecosystem: pandas, scikit‑learn, numpy, imbalanced‑learn, matplotlib.

