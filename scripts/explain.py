import shap
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("creditcard.csv")
X = data.drop(columns=["Class"]).values

# Load model
model = TabNetClassifier()
model.load_model("model.zip")

# SHAP explainer (TabNet has native SHAP-based feature attribution)
explainer = shap.Explainer(model.predict, X[:1000])

# Compute SHAP values for a subset
shap_values = explainer(X[:100])

# Beeswarm plot
shap.plots.beeswarm(shap_values, show=False)
plt.savefig("models/shap_summary.png", bbox_inches='tight')
