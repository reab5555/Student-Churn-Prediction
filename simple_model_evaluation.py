import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from tabulate import tabulate
import json

# Load the data
X_numpy = pd.read_csv("X_resampled.csv").values
y_numpy = pd.read_csv("y_resampled.csv").squeeze().values
X_top10_resampled = pd.read_csv("X_top10_resampled.csv").values
y_top10_resampled = pd.read_csv("y_top10_resampled.csv").squeeze().values

def evaluate_simple_models(X, y, cv=8):
    models = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 20, 25]},
        'KNN': {'n_neighbors': [3, 5, 7, 10, 15, 25, 50, 70, 100]},
        'SVM': {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']},
        'Random Forest': {'n_estimators': [50, 100, 200, 250, 300, 350, 400, 450, 500]},
        'Gradient Boosting': {'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5], 'n_estimators': [25, 50, 100, 150, 200]}
    }

    best_f1_scores = {}
    best_models = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Step 1: Find best parameters using GridSearchCV
        grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='f1', n_jobs=-1)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        # Step 2: Create a new model with the best parameters
        best_model = model.set_params(**best_params)

        # Step 3: Perform cross-validation with the best model
        cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='f1', n_jobs=-1)

        best_f1_scores[name] = cv_scores.mean()
        best_models[name] = best_model

        print(f"Mean F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Step 4: Generate predictions for classification report
        y_pred = cross_val_predict(best_model, X, y, cv=cv)
        print("Classification Report:")
        print(classification_report(y, y_pred))

    return best_f1_scores, best_models


# Evaluate simpler models with all features
print("\nEvaluating simpler models with all features:")
best_f1_scores_all, best_models_all = evaluate_simple_models(X_numpy, y_numpy)

# Save the best F1 scores for all features
with open("best_f1_scores_all.json", "w") as f:
    json.dump(best_f1_scores_all, f)

# Evaluate simpler models with top 10 features
print("\nEvaluating simpler models with top 10 features:")
best_f1_scores_top10, best_models_top10 = evaluate_simple_models(X_top10_resampled, y_top10_resampled)

# Save the best F1 scores for top 10 features
with open("best_f1_scores_top10.json", "w") as f:
    json.dump(best_f1_scores_top10, f)

# Compare F1 scores and determine the best set of features
print("\nComparison of F1 scores:")
print("Model | All Features | Top 10 Features")
for model in best_f1_scores_all.keys():
    print(f"{model} | {best_f1_scores_all[model]:.4f} | {best_f1_scores_top10[model]:.4f}")

# Use the best set of features for neural network evaluation
use_top10 = any(f1 > best_f1_scores_all[name] for name, f1 in best_f1_scores_top10.items())
print(f"\nUsing top 10 features: {use_top10}")
