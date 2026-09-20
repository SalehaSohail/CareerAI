"""
This file contains the semantic embedding logic for CareerAI.

Embeddings convert text into numerical vectors that capture
the semantic meaning of the text.

CareerAI uses these embeddings to compare a user's resume
with job descriptions based on semantic similarity.
"""

from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def generate_embedding(text):
    """
    Convert text into a numerical embedding vector.
    """

    embedding = model.encode(text)

    return embedding


def calculate_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity between two pieces of text.
    """

    embedding1 = generate_embedding(text1)

    embedding2 = generate_embedding(text2)

    similarity = model.similarity(
        embedding1,
        embedding2
    )

    return float(similarity[0][0])


if __name__ == "__main__":

    resume_text = """
    I have experience in Python, machine learning,
    data analysis and building machine learning pipelines.
    """

    job_description = """
    We are looking for a machine learning engineer
    with experience developing predictive models,
    data pipelines and Python-based ML solutions.
    """

    similarity = calculate_semantic_similarity(
        resume_text,
        job_description
    )

    print(
        "Semantic Similarity:",
        round(similarity, 3)
    )