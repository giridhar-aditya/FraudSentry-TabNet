# 🚩 FraudSentry

**A TabNet-based temporal fraud detection framework for transactional data.**

## 📊 Summary

FraudSentry implements a deep learning pipeline for financial fraud detection using the TabNet architecture, optimized for tabular transactional data. The system reflects real-world deployment settings by adopting a temporal validation strategy, where transactions are chronologically split into training and testing sets to mimic production data drift scenarios. Preprocessing is handled through robust scaling to mitigate outlier sensitivity, which is common in financial datasets.

The model leverages TabNet’s attentive feature selection to efficiently capture both linear and non-linear patterns in highly imbalanced datasets. Performance evaluation includes ROC-AUC with bootstrapped confidence intervals for statistical robustness, along with Precision-Recall analysis for better fraud sensitivity measurement. Explainability is supported through SHAP-based feature importance analysis, enabling better model interpretability for high-stakes financial use cases. The trained model is fully serialized for downstream deployment.

## 🔧 Key Components

- **Model:** TabNetClassifier (deep learning for tabular data)
- **Preprocessing:** RobustScaler (outlier-tolerant scaling)
- **Validation:** Temporal split (chronological train/test separation)
- **Evaluation:**  
  - ROC-AUC with bootstrap confidence intervals  
  - Precision-Recall curve  
  - SHAP explainability for feature attribution
- **Deployment-ready:** Full model and scaler serialization (`model.zip` & scaler file)

## 📂 Dataset Requirement

- CSV file (`creditcard.csv`) containing:
  - `Time` — transaction timestamp
  - Feature columns (`V1` to `V28`, `Amount`)
  - `Class` — target label (0: normal, 1: fraud)

## 🚀 Usage

1. Place dataset in `data/` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start training:
    ```bash
    python train.py
    ```
4. Results, metrics, and model files will be saved to the `models/` directory.

## 📈 Future Scope

- Real-time streaming fraud detection integration
- Extended hyperparameter search for production optimization
