"""
CareerAI job ranking and LLM explanation pipeline.

This module:

1. Loads the trained ML relevance model.
2. Extracts the resume text.
3. Calculates resume-job matching once.
4. Uses the trained ML model for relevance scoring.
5. Ranks the jobs.
6. Generates LLM explanations for the top 3 jobs.
"""

import pandas as pd
import joblib

from jobs import get_all_jobs

from resume import (
    extract_text_from_pdf,
    clean_resume_text
)

from matching import calculate_job_matches

from llm_explanation import generate_job_explanation


# -----------------------------
# File paths
# -----------------------------

resume_path = r"D:\CareerAI_Resume\Saleha CV.pdf"

model_path = "src/ranking_model.pkl"


# -----------------------------
# Load trained ML model
# -----------------------------

model = joblib.load(
    model_path
)


# -----------------------------
# Load and process resume
# -----------------------------

raw_resume_text = extract_text_from_pdf(
    resume_path
)

resume_text = clean_resume_text(
    raw_resume_text
)


# -----------------------------
# Load jobs from PostgreSQL
# -----------------------------

jobs = get_all_jobs()


# -----------------------------
# Calculate matching
# ONLY ONCE
# -----------------------------

matches = calculate_job_matches(
    resume_text,
    jobs
)


# -----------------------------
# Convert matching results
# into a DataFrame
# -----------------------------

rows = []

for match in matches:

    job = match["job"]

    rows.append({
        "job_id": job["id"],
        "job_title": job["title"],
        "company": job["company"],
        "tfidf_score": match["tfidf_score"],
        "semantic_score": match["semantic_score"],
        "skill_match": match[
            "skill_match_percentage"
        ],
        "matched_skills": match[
            "matched_skills"
        ],
        "missing_skills": match[
            "missing_skills"
        ]
    })


data = pd.DataFrame(rows)


# -----------------------------
# Prepare features for
# ML relevance model
# -----------------------------

X = data[
    [
        "tfidf_score",
        "semantic_score",
        "skill_match"
    ]
]


# -----------------------------
# Calculate ML relevance score
# -----------------------------

probabilities = model.predict_proba(
    X
)

data["ml_score"] = (
    probabilities[:, 1] * 100
)


# -----------------------------
# Calculate CareerAI
# overall matching score
# -----------------------------

data["final_score"] = (
    0.40 * data["tfidf_score"]
    + 0.30 * (
        data["skill_match"] / 100
    )
    + 0.30 * data["semantic_score"]
) * 100


# -----------------------------
# Rank jobs
# -----------------------------

data = data.sort_values(
    "final_score",
    ascending=False
).reset_index(drop=True)


# -----------------------------
# Display ranking
# -----------------------------

print("\nCareerAI Job Ranking")
print("=" * 60)


for index, row in data.iterrows():

    print(
        f"\nRank: {index + 1}"
    )

    print(
        f"Job: {row['job_title']}"
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
        f"ML Relevance Score: "
        f"{row['ml_score']:.2f}%"
    )

    print(
        f"Final Match Score: "
        f"{row['final_score']:.2f}%"
    )


# -----------------------------
# Generate LLM explanations
# for the top 3 jobs
# -----------------------------

print("\n\nCareerAI AI Explanations")
print("=" * 60)


top_jobs = data.head(3)


for index, row in top_jobs.iterrows():

    explanation = generate_job_explanation(
        job_title=row["job_title"],
        company=row["company"],
        skill_match=row["skill_match"],
        matched_skills=row[
            "matched_skills"
        ],
        missing_skills=row[
            "missing_skills"
        ],
        tfidf_score=row[
            "tfidf_score"
        ],
        semantic_score=row[
            "semantic_score"
        ]
    )

    print(
        f"\nRank {index + 1}: "
        f"{row['job_title']} — "
        f"{row['company']}"
    )

    print("-" * 60)

    print(explanation)