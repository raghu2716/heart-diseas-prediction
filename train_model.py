# train_model.py
# Trains the heart disease model and saves it as a pickle file.

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
heart = pd.read_csv("heart_cleveland_upload.csv")

# Make a copy
heart_df = heart.copy()

# Rename condition -> target if needed
if "condition" in heart_df.columns and "target" not in heart_df.columns:
    heart_df = heart_df.rename(columns={"condition": "target"})

print("Columns:", heart_df.columns.tolist())

# 2. Split into features and target
X = heart_df.drop(columns="target")
y = heart_df["target"]

# 3. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Create pipeline: scaler + RandomForest
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]
)

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy: {:.2f}%".format(acc * 100))
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 7. Save the pipeline (scaler + model) to pickle
filename = "heart-disease-prediction-knn-model.pkl"
with open(filename, "wb") as f:
    pickle.dump(model, f)

print(f"Saved trained model pipeline to {filename}")
