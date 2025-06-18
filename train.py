import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Load dataset
data = pd.read_csv("creditcard.csv")
X = data.drop(columns=["Class"]).values
y = data["Class"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Create directory to save model
os.makedirs("models", exist_ok=True)

# Initialize TabNetClassifier
clf = TabNetClassifier(
    n_d=64,
    n_a=64,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="entmax"
)

print("Training TabNet model...")
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=['auc'],
    patience=10,
    max_epochs=100,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=4,
    drop_last=False
)

# Save trained model
clf.save_model("models/model")
print("Model successfully saved.")

# Evaluate on validation set
preds_val = clf.predict(X_val)
probs_val = clf.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, probs_val)

print("\nValidation Metrics:")
print(classification_report(y_val, preds_val, digits=4))
print(f"Validation AUC: {auc_val:.5f}")
