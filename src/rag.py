"""
RAG retrieval module for CareerAI.

This module retrieves the most relevant job documents
from the CareerAI PostgreSQL database based on a query.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rag_documents import create_job_documents


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CREATE DOCUMENT EMBEDDINGS
# ============================================================

def create_document_embeddings(documents):
    """
    Convert all job documents into semantic embeddings.
    """

    embeddings = embedding_model.encode(
        documents
    )

    return embeddings


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    documents,
    document_embeddings,
    top_k=3
):
    """
    Retrieve the most relevant job documents
    for a given query.
    """

    query_embedding = embedding_model.encode(
        [query]
    )

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "document": documents[index],
                "similarity": float(
                    similarities[index]
                )
            }
        )

    return results


# ============================================================
# TEST RAG
# ============================================================

if __name__ == "__main__":

    # Get real jobs from PostgreSQL
    documents = create_job_documents()

    print(
        f"\nTotal RAG documents: {len(documents)}"
    )

    # Create embeddings
    document_embeddings = create_document_embeddings(
        documents
    )

    # Example search query
    query = """
    I want a junior AI or machine learning job
    involving Python, machine learning, NLP,
    LLMs and FastAPI.
    """

    # Retrieve top 3 jobs
    results = retrieve_documents(
        query=query,
        documents=documents,
        document_embeddings=document_embeddings,
        top_k=3
    )

    print("\nRAG Retrieval Results")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            "Similarity:",
            round(
                result["similarity"],
                3
            )
        )

        print(
            "Document:"
        )

        print(
            result["document"]
        )

        print(
            "-" * 60
        )