"""
Create a larger prototype training dataset for CareerAI.

The dataset contains multiple resume profiles matched against
the available jobs.

Each resume-job pair contains:

- TF-IDF similarity
- Semantic similarity
- Skill match percentage
- Label

Label:
1 = relevant match
0 = not a relevant match
"""

import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import generate_embedding
from jobs import get_all_jobs
from skill_matching import calculate_skill_match


# ------------------------------------------------
# Prototype resume profiles
# ------------------------------------------------

resumes = {

    "ML_Profile": """
    Computer Science graduate with experience in Python,
    machine learning, Pandas, NumPy, Scikit-learn,
    PyTorch, feature engineering, data preprocessing,
    model evaluation, classification, regression,
    SQL and machine learning pipelines.
    """,

    "Data_Profile": """
    Computer Science graduate with experience in Python,
    SQL, PostgreSQL, Pandas, NumPy, statistics,
    exploratory data analysis, data visualization,
    data analysis, feature engineering and predictive modeling.
    """,

    "AI_Profile": """
    Computer Science graduate with experience in Python,
    machine learning, NLP, LLMs, REST APIs, FastAPI,
    embeddings, prompt engineering, RAG, API integration
    and Git.
    """,

    "Backend_Profile": """
    Computer Science graduate with experience in Python,
    object oriented programming, SQL, PostgreSQL,
    FastAPI, REST APIs, authentication, testing,
    Docker, backend development and Git.
    """,

    "NLP_Profile": """
    Computer Science graduate with experience in Python,
    machine learning, NLP, text processing, tokenization,
    embeddings, PyTorch, Scikit-learn and language models.
    """
}


# ------------------------------------------------
# Labels for each resume-job pair
# ------------------------------------------------

labels = {

    "ML_Profile": {
        "Junior Machine Learning Engineer": 1,
        "Junior Data Scientist": 1,
        "AI Engineer": 0,
        "Python Developer": 0,
        "Machine Learning Intern": 1,
        "Data Analyst": 1,
        "Junior AI Developer": 0,
        "NLP Engineer": 0,
        "Backend Python Developer": 0,
        "Junior AI/ML Engineer": 1
    },

    "Data_Profile": {
        "Junior Machine Learning Engineer": 1,
        "Junior Data Scientist": 1,
        "AI Engineer": 0,
        "Python Developer": 0,
        "Machine Learning Intern": 1,
        "Data Analyst": 1,
        "Junior AI Developer": 0,
        "NLP Engineer": 0,
        "Backend Python Developer": 0,
        "Junior AI/ML Engineer": 1
    },

    "AI_Profile": {
        "Junior Machine Learning Engineer": 0,
        "Junior Data Scientist": 0,
        "AI Engineer": 1,
        "Python Developer": 0,
        "Machine Learning Intern": 0,
        "Data Analyst": 0,
        "Junior AI Developer": 1,
        "NLP Engineer": 1,
        "Backend Python Developer": 1,
        "Junior AI/ML Engineer": 1
    },

    "Backend_Profile": {
        "Junior Machine Learning Engineer": 0,
        "Junior Data Scientist": 0,
        "AI Engineer": 1,
        "Python Developer": 1,
        "Machine Learning Intern": 0,
        "Data Analyst": 1,
        "Junior AI Developer": 1,
        "NLP Engineer": 0,
        "Backend Python Developer": 1,
        "Junior AI/ML Engineer": 0
    },

    "NLP_Profile": {
        "Junior Machine Learning Engineer": 0,
        "Junior Data Scientist": 0,
        "AI Engineer": 1,
        "Python Developer": 0,
        "Machine Learning Intern": 1,
        "Data Analyst": 0,
        "Junior AI Developer": 1,
        "NLP Engineer": 1,
        "Backend Python Developer": 0,
        "Junior AI/ML Engineer": 1
    }
}


# ------------------------------------------------
# Get jobs from PostgreSQL
# ------------------------------------------------

jobs = get_all_jobs()


rows = []


# ------------------------------------------------
# Create features for every resume-job pair
# ------------------------------------------------

for resume_name, resume_text in resumes.items():

    job_descriptions = []

    for job in jobs:
        job_descriptions.append(
            job["description"]
        )

    documents = [resume_text] + job_descriptions

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    resume_vector = tfidf_matrix[0]

    job_vectors = tfidf_matrix[1:]

    tfidf_scores = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]


    resume_embedding = generate_embedding(
        resume_text
    )


    for job, tfidf_score in zip(
        jobs,
        tfidf_scores
    ):

        job_embedding = generate_embedding(
            job["description"]
        )

        semantic_score = cosine_similarity(
            [resume_embedding],
            [job_embedding]
        )[0][0]


        skill_result = calculate_skill_match(
            resume_text,
            job["required_skills"]
        )


        rows.append({
            "resume_name": resume_name,
            "job_title": job["title"],
            "company": job["company"],
            "tfidf_score": round(
                float(tfidf_score),
                4
            ),
            "semantic_score": round(
                float(semantic_score),
                4
            ),
            "skill_match": skill_result[
                "skill_match_percentage"
            ],
            "label": labels[
                resume_name
            ][
                job["title"]
            ]
        })


# ------------------------------------------------
# Save training dataset
# ------------------------------------------------

output_path = "data/training_data.csv"


with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "resume_name",
            "job_title",
            "company",
            "tfidf_score",
            "semantic_score",
            "skill_match",
            "label"
        ]
    )

    writer.writeheader()

    writer.writerows(rows)


print(
    f"Training dataset created: {output_path}"
)

print(
    f"Total training examples: {len(rows)}"
)