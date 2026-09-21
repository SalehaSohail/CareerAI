
"""
Train and evaluate the CareerAI job matching model.

The model uses:

- TF-IDF similarity
- Semantic similarity
- Skill match percentage

to predict whether a resume-job pair
is a relevant match.
"""

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Load training dataset
# -----------------------------

data = pd.read_csv(
    "data/training_data.csv"
)


# -----------------------------
# Select input features
# -----------------------------

X = data[
    [
        "tfidf_score",
        "semantic_score",
        "skill_match"
    ]
]


# -----------------------------
# Select target label
# -----------------------------

y = data["label"]


# -----------------------------
# Split the dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Create the ML model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -----------------------------
# Train the model
# -----------------------------

model.fit(
    X_train,
    y_train
)


# -----------------------------
# Evaluate the model
# -----------------------------

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    f"Test Accuracy: {accuracy:.2f}"
)


print(
    "\nClassification Report:"
)


print(
    classification_report(
        y_test,
        predictions
    )
)


# -----------------------------
# Train final model
# -----------------------------

model.fit(
    X,
    y
)


# -----------------------------
# Feature importance
# -----------------------------

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})


feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)


print(
    "\nFeature Importance:"
)


print(
    feature_importance.to_string(
        index=False
    )
)


# -----------------------------
# Save final model
# -----------------------------

joblib.dump(
    model,
    "src/ranking_model.pkl"
)


print(
    "\nFinal ranking model trained and saved successfully!"
)



