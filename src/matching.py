"""
This file contains the job-matching logic for CareerAI.

CareerAI uses three matching signals:

1. TF-IDF similarity
   Measures similarity based on important words.

2. Skill matching
   Checks which required job skills are present in the resume.

3. Semantic similarity
   Uses sentence embeddings to compare the meaning of the
   resume and job description.

These signals are combined to produce a final matching score.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skill_matching import calculate_skill_match
from embeddings import generate_embedding


def calculate_job_matches(resume_text, jobs):
    """
    Compare the resume with every job using:

    - TF-IDF similarity
    - Skill matching
    - Semantic embeddings
    - Combined final score
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
    # Semantic embeddings
    # -----------------------------

    resume_embedding = generate_embedding(
        resume_text
    )

    job_embeddings = []

    for description in job_descriptions:
        embedding = generate_embedding(
            description
        )

        job_embeddings.append(embedding)

    semantic_scores = []

    for job_embedding in job_embeddings:

        similarity = cosine_similarity(
            [resume_embedding],
            [job_embedding]
        )[0][0]

        semantic_scores.append(
            float(similarity)
        )

    # -----------------------------
    # Combine all matching signals
    # -----------------------------

    results = []

    for job, tfidf_score, semantic_score in zip(
        jobs,
        tfidf_scores,
        semantic_scores
    ):

        skill_result = calculate_skill_match(
            resume_text,
            job["required_skills"]
        )

        skill_score = (
            skill_result["skill_match_percentage"] / 100
        )

        final_score = (
            0.40 * float(tfidf_score)
            + 0.30 * skill_score
            + 0.30 * semantic_score
        ) * 100

        result = {
            "job": job,
            "tfidf_score": float(tfidf_score),
            "semantic_score": semantic_score,
            "skill_match_percentage": skill_result[
                "skill_match_percentage"
            ],
            "final_score": final_score,
            "matched_skills": skill_result[
                "matched_skills"
            ],
            "missing_skills": skill_result[
                "missing_skills"
            ]
        }

        results.append(result)

    # Sort jobs according to final matching score

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    return results


if __name__ == "__main__":

    from jobs import get_all_jobs
    from resume import (
        extract_text_from_pdf,
        clean_resume_text
    )

    resume_path = r"D:\CareerAI_Resume\Saleha CV.pdf"

    raw_resume_text = extract_text_from_pdf(
        resume_path
    )

    resume_text = clean_resume_text(
        raw_resume_text
    )

    jobs = get_all_jobs()

    matches = calculate_job_matches(
        resume_text,
        jobs
    )

    for match in matches:

        print(
            "\n" + match["job"]["title"]
        )

        print(
            "TF-IDF Score:",
            round(match["tfidf_score"], 3)
        )

        print(
            "Semantic Score:",
            round(match["semantic_score"], 3)
        )

        print(
            "Skill Match:",
            match["skill_match_percentage"],
            "%"
        )

        print(
            "Final Score:",
            round(match["final_score"], 2),
            "%"
        )

        print(
            "Matched Skills:",
            ", ".join(
                match["matched_skills"]
            )
        )

        print(
            "Missing Skills:",
            ", ".join(
                match["missing_skills"]
            )
        )