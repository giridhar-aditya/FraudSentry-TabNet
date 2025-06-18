import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_balanced_data(data, balance=True):
    """
    Load dataset with optional class balancing.
    
    Parameters:
        data (pd.DataFrame): Input dataset containing features and 'Class' label.
        balance (bool): Whether to balance classes via undersampling.

    Returns:
        X (np.array): Feature matrix.
        y (np.array): Target labels.
    """
    if balance:
        class_1 = data[data['Class'] == 1]
        class_0 = data[data['Class'] == 0].sample(n=len(class_1), random_state=42)
        balanced_data = pd.concat([class_0, class_1])
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        X = balanced_data.drop(columns=["Class"]).values
        y = balanced_data["Class"].values
    else:
        X = data.drop(columns=["Class"]).values
        y = data["Class"].values
    return X, y

def evaluate_model(model_path, data_path="creditcard.csv", balance=False, threshold=0.5):
    """
    Evaluate a trained TabNet model on the dataset.

    Parameters:
        model_path (str): Path to the saved TabNet model file.
        data_path (str): Path to the CSV dataset file.
        balance (bool): Whether to balance classes before evaluation.
        threshold (float): Classification threshold for converting probabilities to classes.
    """
    # Load dataset
    data = pd.read_csv(data_path)
    X, y = load_balanced_data(data, balance=balance)

    # Load trained TabNet model
    model = TabNetClassifier()
    model.load_model(model_path)

    # Predict probabilities and convert to class labels
    pred_probs = model.predict_proba(X)[:, 1]
    pred_classes = (pred_probs > threshold).astype(int)

    # Display evaluation metrics
    print("\n--- Evaluation Report ---")
    print("Balanced Test" if balance else "Full Dataset Test")
    print("Classification Report:")
    print(classification_report(y, pred_classes, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y, pred_classes))
    auc = roc_auc_score(y, pred_probs)
    print(f"ROC AUC Score: {auc:.5f}")
    print("--- End of Report ---\n")

# ===== Run evaluation =====
# Set 'balance' to True for balanced evaluation or False for full dataset evaluation.

evaluate_model(
    model_path="model.zip",
    data_path="creditcard.csv",
    balance=True,   # <-- Toggle this flag
    threshold=0.5
)
