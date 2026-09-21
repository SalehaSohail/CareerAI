"""
Use the trained CareerAI model to rank jobs.

The model uses:

- TF-IDF similarity
- Semantic similarity
- Skill match percentage
"""

import joblib
import pandas as pd


# -----------------------------
# Load the trained model
# -----------------------------

model = joblib.load(
    "src/ranking_model.pkl"
)


# -----------------------------
# Load the matching dataset
# -----------------------------

data = pd.read_csv(
    "data/matching_data.csv"
)


# -----------------------------
# Select matching features
# -----------------------------

X = data[
    [
        "tfidf_score",
        "semantic_score",
        "skill_match"
    ]
]


# -----------------------------
# Get ML predictions
# -----------------------------

predictions = model.predict_proba(X)


# -----------------------------
# Add ML score
# -----------------------------

data["ml_score"] = (
    predictions[:, 1] * 100
)


# -----------------------------
# Sort jobs by ML score
# -----------------------------

data = data.sort_values(
    "ml_score",
    ascending=False
)


# -----------------------------
# Display ranked jobs
# -----------------------------

print("\nCareerAI Job Ranking")
print("=" * 50)


for _, row in data.iterrows():

    print(
        f"\nJob: {row['job_title']}"
    )

    print(
        f"Company: {row['company']}"
    )

    print(
        f"TF-IDF Score: "
        f"{row['tfidf_score']:.4f}"
    )

    print(
        f"Semantic Score: "
        f"{row['semantic_score']:.4f}"
    )

    print(
        f"Skill Match: "
        f"{row['skill_match']:.2f}%"
    )

    print(
        f"ML Score: "
        f"{row['ml_score']:.2f}%"
    )