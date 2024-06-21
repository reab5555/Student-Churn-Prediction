import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier

# Load the data
X = pd.read_csv("X_resampled.csv")
y = pd.read_csv("y_resampled.csv").squeeze()

# Load the best F1 scores from evaluation scripts
with open("best_f1_scores_all.json", "r") as f:
    best_f1_scores_all = json.load(f)

with open("best_f1_scores_top10.json", "r") as f:
    best_f1_scores_top10 = json.load(f)

# Train a Random Forest model
rf = RandomForestClassifier()
rf.fit(X, y)

# Print and sort the features used for the best model from most important to least
if hasattr(rf, 'feature_importances_'):
    feature_importances = rf.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = X.columns[sorted_indices]
    sorted_importances = feature_importances[sorted_indices]
    print("\nFeature importance for the best model (Random Forest):")
    for feature, importance in zip(sorted_features, sorted_importances):
        print(f"{feature}: {importance:.4f}")

# Check if the best configuration is using top 10 features
use_top10 = any(f1 > best_f1_scores_all[name] for name, f1 in best_f1_scores_top10.items())
print(f"Best configuration is using top 10 features: {use_top10}")
