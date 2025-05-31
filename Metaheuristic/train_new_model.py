import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv("perfectly_balanced_dataset.csv")

# Define features in exact order
feature_cols = [
    'num_tasks',
    'avg_execution',
    'avg_priority',
    'avg_deadline',
    'periodic_ratio',
    'utilization'
]

X = df[feature_cols]
y = df['scheduler']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Save artifacts
print("\nSaving model artifacts...")
joblib.dump(model, 'new_model.joblib')
joblib.dump(scaler, 'new_scaler.joblib')
joblib.dump(le, 'new_encoder.joblib')

# Print model information
print("\nFeature names used in training:", feature_cols)
print("Unique schedulers:", list(le.classes_))
print("Model score on test set:", model.score(X_test_scaled, y_test))
print("Done!") 