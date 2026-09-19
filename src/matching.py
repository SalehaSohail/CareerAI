
"""
This file contains the first draft of the job matching algorithm of CareerAI.

In the current state, this module compares resume text against
job descriptions found in the database stored in PostgreSQL
and calculates how similar the texts are. In the current state,
CareerAI uses TF-IDF (Term Frequency-Inverse Document
Frequency) as the algorithm for transforming text into numeric
vectors and cosine similarity to measure the degree of their
similarity.

The matching function is separated from the database function
to give each of them its responsibilities. The database function
fetches jobs from the database, while the matching function
does text comparison based on machine learning algorithms.

This is the first draft of the matching algorithm. Further
development may involve semantic embeddings, machine learning
ranking, LLM explanation, and RAG.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_job_matches(resume_text, jobs):
    """
    Compare resume text with job descriptions and calculate
    a similarity score for every job.
    """

    job_descriptions = []

    for job in jobs:
        job_descriptions.append(job["description"])

    documents = [resume_text] + job_descriptions

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    resume_vector = tfidf_matrix[0]
    job_vectors = tfidf_matrix[1:]

    similarity_scores = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]

    results = []

    for job, score in zip(jobs, similarity_scores):
        result = {
            "job": job,
            "score": float(score)
        }

        results.append(result)

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results


if __name__ == "__main__":
    from jobs import get_all_jobs

    resume_text = """
    Python SQL machine learning pandas scikit-learn FastAPI
    data analysis artificial intelligence
    """

    jobs = get_all_jobs()

    matches = calculate_job_matches(resume_text, jobs)

    for match in matches:
        print(
            match["job"]["title"],
            "->",
            round(match["score"], 3)
        )