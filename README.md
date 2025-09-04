# Healthcare Provider Fraud Detection (Medicare Claims)

End‑to‑end machine‑learning pipeline to flag **potentially fraudulent healthcare providers** from Medicare claim patterns. The repository is **code‑first (Python)** and all **CSVs in this repo are program‑generated outputs** (processed datasets and submissions) — not the original raw Kaggle files.

---

## 🚀 TL;DR
- **Goal:** Predict `PotentialFraud` at the **provider** level using inpatient/outpatient/beneficiary signals.
- **Approach:** Clean & merge → feature engineering → class‑imbalance handling → train multiple models → choose operating threshold → persist artifacts and outputs.
- **Artifacts:** Trained models (`.joblib`), metadata (`.json`), and **CSV outputs saved by the script** for reproducibility.

---

## 📊 Dataset
- **Source:** Kaggle — [Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data)
- **Kaggle link (provided):** https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data
- **Tables:** Beneficiary, Inpatient, Outpatient, and Provider label table.
- **Target:** `PotentialFraud` (Yes/No) at provider level.

> 🔎 **Note:** Raw Kaggle CSVs are **not** committed. The repository contains **processed CSVs** and **submission CSVs** that are **saved automatically by the program** during runs.

---

## 🗂️ What’s in this Repo
Actual file names may vary across runs, but you will typically see:

```
├─ healthcare_fraud_detection.py     # Main pipeline script (code-first project)
├─ models/                           # Trained models + transformers + metadata
│  ├─ logistic_regression_threshold_60.joblib
│  ├─ random_forest.joblib
│  ├─ isolation_forest.joblib
│  ├─ isolation_forest_pca_scaler.joblib
│  ├─ isolation_forest_pca_transformer.joblib
│  ├─ logistic_regression_scaler.joblib
│  ├─ *metadata.json                 # feature list, thresholds, versions
│  └─ ...
├─ data/
│  ├─ processed/                     # Program-saved processed datasets (features, joins)
│  └─ submissions/                   # Program-saved prediction CSVs (see below)
│     ├─ Submission_logistic_regression_threshold_60.csv
│     ├─ Submission_Random_Forest_Classifier.csv
│     └─ Submission_Isolation_Forest.csv
├─ README.md
└─ requirements.txt
```

> If you see these files at the **repo root** currently, that’s fine functionally. For a more professional layout, consider placing them under the `models/` and `data/` folders as shown above.

---

## ⚙️ Setup
```bash
# 1) Create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```
**Suggested requirements (edit if your versions differ):**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.11
matplotlib>=3.7
joblib>=1.3
```

---

## 🧪 How to Run the Pipeline
> The project is implemented as a single Python program. Paths and options are defined in the script; some runs may expose CLI flags, but the default is a one‑command run.

### Option A — One command
```bash
python healthcare_fraud_detection.py
```
This will:
- Load/prepare data (from your configured paths)
- Engineer provider‑level features
- Train models (Logistic Regression, Random Forest, Isolation Forest)
- Apply thresholding (e.g., 0.60 for the LR variant in this run)
- **Save processed datasets and predictions as CSVs** under `data/processed/` and `data/submissions/`
- Persist trained models + scalers + metadata to `models/`

### Option B — With explicit arguments (if enabled)
```bash
python healthcare_fraud_detection.py \
  --raw_dir data/raw \
  --processed_dir data/processed \
  --models_dir models \
  --submissions_dir data/submissions \
  --model lr --threshold 0.60 --seed 42
```
> If the above flags are not implemented in your local copy, use **Option A** and edit the configuration constants at the top of the script.

---

## 📦 Outputs (Saved by the Program)
- **Models & Transformers** (`models/`)
  - `logistic_regression_threshold_60.joblib`
  - `random_forest.joblib`
  - `isolation_forest.joblib`
  - `logistic_regression_scaler.joblib`
  - `isolation_forest_pca_scaler.joblib`, `isolation_forest_pca_transformer.joblib`
  - `*_metadata.json` (feature order, preprocessing config, thresholds, versions)
- **Program‑Generated CSVs**
  - `data/processed/*.csv` — cleaned/merged/engineered datasets at provider level
  - `data/submissions/Submission_*.csv` — predictions for evaluation/submit

> ✅ These CSVs are included in the repo intentionally to make the run **reproducible** for reviewers without re‑processing the raw Kaggle files.

---

## 🏗️ Methodology (High‑Level)
1. **Cleaning & Integration** — Join Beneficiary, Inpatient, Outpatient claims → provider‑level table.
2. **Feature Engineering** — Claim counts, unique beneficiaries, reimbursement and deductible stats, LOS metrics, procedure/diagnosis diversity, temporal patterns.
3. **Imbalance Handling** — Class weights and/or sampling where appropriate.
4. **Modeling** —
   - **Logistic Regression** (with scaler)
   - **Random Forest Classifier**
   - **Isolation Forest** (unsupervised anomaly signals, optionally with PCA)
5. **Evaluation** — Accuracy, Precision, Recall, F1, ROC‑AUC; select threshold to favor recall at acceptable precision.
6. **Persistence** — Save models, scalers, metadata; export processed datasets and submissions as CSVs.

---

## 🔮 Inference Example (Load a Saved Model)
```python
import joblib
import pandas as pd

# Provider-level features must match the training feature list & order
X = pd.read_csv("data/processed/provider_features.csv")
clf = joblib.load("models/logistic_regression_threshold_60.joblib")
proba = clf.predict_proba(X)[:, 1]
X.assign(fraud_risk=proba).to_csv("data/submissions/predictions_lr.csv", index=False)
```
> Always keep the **feature order** consistent with the order stored in your `*_metadata.json`.

---

## 🔁 Reproducibility & Notes
- Fixed random seed where applicable (default `42`).
- `requirements.txt` captures core dependencies; pin exact versions for strict reproducibility.
- Large artifacts (>100 MB) should use **Git LFS** or external storage.

---

## 🧭 Responsible Use
This project flags **potential** fraud for investigation. It **does not** determine guilt. Any deployment should include human oversight, audit logs, bias checks, and policy‑aligned thresholds.

---

## 🙌 Acknowledgments
- Kaggle dataset: *Healthcare Provider Fraud Detection Analysis*.
- Python ecosystem: pandas, scikit‑learn, numpy, imbalanced‑learn, matplotlib.

---

## 🗺️ Roadmap (Optional)
- [ ] Move files into `models/` and `data/…` folders for a cleaner structure
- [ ] Add a tiny `demo.ipynb` (cleared outputs) for quick walkthrough
- [ ] Add CLI flags/config file for reproducible experiments
- [ ] Add unit tests for data prep & feature builders
