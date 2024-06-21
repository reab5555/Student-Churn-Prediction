import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.under_sampling import RandomUnderSampler

# Load the data
X = pd.read_csv("X_resampled.csv")
y = pd.read_csv("y_resampled.csv").squeeze()

# Ensure X and y have the same number of samples
if len(X) != len(y):
    raise ValueError(f"Inconsistent number of samples: X has {len(X)}, y has {len(y)}")

# Select top 10 features using SelectKBest
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)
top_features = X.columns[selector.get_support(indices=True)]

print('Selected Features:', top_features)

# Resample top 10 features to match y_resampled
rus = RandomUnderSampler(random_state=42)
X_top10_resampled, y_top10_resampled = rus.fit_resample(X_new, y)

# Save the selected features data
pd.DataFrame(X_top10_resampled, columns=top_features).to_csv("X_top10_resampled.csv", index=False)
pd.Series(y_top10_resampled).to_csv("y_top10_resampled.csv", index=False)
