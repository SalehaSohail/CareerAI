"""
Create a dataset containing resume-job matching features.

The dataset will be used later to experiment with a
machine learning ranking model.

Each row represents one resume-job pair and contains:

- TF-IDF similarity
- Semantic similarity
- Skill match percentage
"""

import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import generate_embedding
from jobs import get_all_jobs
from resume import extract_text_from_pdf, clean_resume_text
from skill_matching import calculate_skill_match


def create_matching_dataset(resume_text, jobs):
    """
    Calculate matching features for every job.
    """

    job_descriptions = []

    for job in jobs:
        job_descriptions.append(job["description"])

    # -----------------------------
    # TF-IDF similarity
    # -----------------------------

    documents = [resume_text] + job_descriptions

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    resume_vector = tfidf_matrix[0]

    job_vectors = tfidf_matrix[1:]

    tfidf_scores = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]

    # -----------------------------
    # Resume embedding
    # -----------------------------

    resume_embedding = generate_embedding(
        resume_text
    )

    rows = []

    for job, tfidf_score in zip(
        jobs,
        tfidf_scores
    ):

        # Semantic similarity

        job_embedding = generate_embedding(
            job["description"]
        )

        semantic_score = cosine_similarity(
            [resume_embedding],
            [job_embedding]
        )[0][0]

        # Skill matching

        skill_result = calculate_skill_match(
            resume_text,
            job["required_skills"]
        )

        rows.append({
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
            ]
        })

    return rows


if __name__ == "__main__":

    resume_path = r"D:\CareerAI_Resume\Saleha CV.pdf"

    raw_resume_text = extract_text_from_pdf(
        resume_path
    )

    resume_text = clean_resume_text(
        raw_resume_text
    )

    jobs = get_all_jobs()

    rows = create_matching_dataset(
        resume_text,
        jobs
    )

    output_path = "data/matching_data.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "job_title",
                "company",
                "tfidf_score",
                "semantic_score",
                "skill_match"
            ]
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"Matching dataset created: {output_path}"
    )