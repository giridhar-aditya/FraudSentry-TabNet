# 🚩 FraudSentry

**A TabNet-based temporal fraud detection framework for transactional data.**

---

## 📊 Summary

FraudSentry implements a deep learning pipeline for financial fraud detection using the TabNet architecture, specifically designed for tabular transactional data. The framework simulates real-world deployment conditions by using **temporal validation**, where transactions are chronologically split into training and test sets to reflect natural data drift over time.

The dataset used is highly **imbalanced**, with fraudulent transactions being extremely rare (less than 0.2% of total data). To address this, the system supports:

- **Full Dataset Mode** (retains real-world class imbalance)
- **Balanced Dataset Mode** (undersamples majority class for evaluation)

Preprocessing is performed using **RobustScaler** to minimize the influence of outliers, which are common in financial transaction amounts.

The model leverages TabNet’s **attentive feature selection** to automatically identify important features without manual feature engineering. Performance is evaluated using:

- **ROC-AUC with bootstrapped confidence intervals**
- **Precision-Recall curves** (more informative for imbalanced datasets)

For model interpretability, **SHAP-based feature importance** is integrated to explain predictions — crucial for regulated financial systems.

The entire model pipeline (model + scaler) is fully serialized for easy deployment.

---

## 📂 Dataset Information

We use the publicly available **Credit Card Fraud Detection Dataset**:

**🔗 [Download Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

### Dataset Columns:

- `Time` — time in seconds since the first transaction
- `V1` to `V28` — PCA-transformed anonymized features
- `Amount` — transaction amount
- `Class` — target label (`0`: legitimate, `1`: fraud)

> ⚠ **Imbalance Notice**  
> Only ~0.17% of the transactions are fraudulent. This makes it a challenging but highly realistic fraud detection scenario. We handle this using two evaluation modes:

- **Full Dataset Mode:**  
  Keeps original class imbalance. Used for realistic evaluation reflecting production environment.
- **Balanced Dataset Mode:**  
  Random undersampling of legitimate transactions to match fraud count. Used for stress-testing model sensitivity.

---

## 🔧 Key Components

- **Model:** `TabNetClassifier` (deep learning for tabular data)
- **Preprocessing:** `RobustScaler` (outlier-tolerant feature scaling)
- **Validation:** Temporal train/test split (chronological separation)
- **Evaluation:**
  - ROC-AUC (with bootstrap confidence intervals)
  - Precision-Recall Curve
- **Explainability:** SHAP feature attribution
- **Deployment:** Model and scaler fully serialized (`models/model.zip`, `models/scaler.pkl`)

---
