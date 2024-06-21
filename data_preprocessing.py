import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Load the data (assuming it's in a CSV file)
data = pd.read_csv(r"students_dataset.csv")

# Separate features and target
X = data.drop(['Target'], axis=1)
y = data['Target']

# Print the number of samples for each class
print("Class distribution before resampling:")
print(y.value_counts())

# Define feature types
numeric_features = ['Age at enrollment', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
                    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
                    'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)',
                    'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
                    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (without evaluations)',
                    'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']

binary_features = ['Gender', 'Daytime/evening attendance', 'Educational special needs', 'Scholarship holder', 'International']

categorical_features = ['Marital status', 'Course', 'Previous qualification', 'Mother qualification', 'Father qualification',
                        'Mother occupation', 'Father occupation']

# Encode categorical features
X = pd.get_dummies(X, columns=categorical_features)

# Encode target
status_mapping = {'Dropout': 0, 'Graduate': 1}
y = y.map(status_mapping)

# Scale numerical features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Undersample the majority class if classes are uneven
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Print the number of samples for each class after resampling
print("Class distribution after resampling:")
print(pd.Series(y_resampled).value_counts())

# Convert all columns to float
X_resampled = X_resampled.astype(float)

# Keep data as numpy arrays for splitting
X_numpy = X_resampled.values
y_numpy = y_resampled.values

# Save the processed data
X_resampled.to_csv("X_resampled.csv", index=False)
pd.Series(y_resampled).to_csv("y_resampled.csv", index=False)
