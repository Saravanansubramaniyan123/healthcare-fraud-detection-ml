## 🩺 Problem Statement
Healthcare fraud is a serious issue in the insurance industry, where some providers submit false or misleading claims to obtain unwarranted reimbursements. This leads to billions of dollars in financial losses and impacts patients who genuinely need care.

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

## 📈 Impact
This project demonstrates how machine learning can assist insurers in **flagging suspicious providers early**, reducing fraudulent reimbursements, and supporting human investigators with data-driven insights.


