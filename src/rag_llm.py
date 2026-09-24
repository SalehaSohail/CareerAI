"""
RAG + LLM explanation module for CareerAI.

This module takes the jobs retrieved by the RAG system
and sends their context to an LLM.

The LLM uses only the retrieved job information to
generate a grounded explanation.
"""

import os

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def generate_rag_explanation(query, retrieved_documents):
    """
    Generate an AI explanation using retrieved RAG documents.
    """

    context = ""

    for rank, result in enumerate(
        retrieved_documents,
        start=1
    ):
        context += f"""
--- Retrieved Job {rank} ---

Similarity:
{result["similarity"]:.4f}

Job Information:
{result["document"]}

"""

    prompt = f"""
You are an AI career assistant inside CareerAI.

The user has provided the following query:

{query}

Below is information retrieved by CareerAI's
RAG system from its job database.

Use ONLY this retrieved information.

IMPORTANT RULES:

- Do not invent job information.
- Do not invent skills or requirements.
- Do not invent company information.
- Do not create hiring predictions.
- Do not say the user will get a job.
- Explain the retrieved jobs based only on the provided context.
- Clearly mention relevant skills and requirements when available.

Retrieved Context:
{context}

Provide a concise response using these sections:

1. Relevant Jobs

Briefly explain which retrieved jobs are relevant to the query.

2. Why They Are Relevant

Explain the connection between the user's query
and the retrieved job requirements.

3. Skills to Focus On

Mention the important skills appearing in the
retrieved job information.

Keep the explanation practical and concise.
"""

    try:

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.choices[0].message.content

        if not content:
            return (
                "AI explanation is temporarily unavailable."
            )

        if content.strip().lower() == "user safety: safe":
            return (
                "AI explanation is temporarily unavailable."
            )

        return content

    except Exception as error:

        print(
            "RAG LLM Error:",
            error
        )

        return (
            "AI explanation is temporarily unavailable. "
            "The RAG retrieval results are still available."
        )
if __name__ == "__main__":

    from rag import (
        create_document_embeddings,
        retrieve_documents
    )

    from rag_documents import create_job_documents


    query = """
    I want a junior AI or machine learning job
    involving Python, NLP, LLMs and FastAPI.
    """


    documents = create_job_documents()

    document_embeddings = create_document_embeddings(
        documents
    )


    retrieved_documents = retrieve_documents(
        query=query,
        documents=documents,
        document_embeddings=document_embeddings,
        top_k=3
    )


    print("\nRAG Retrieved Jobs")
    print("=" * 60)

    for rank, result in enumerate(
        retrieved_documents,
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
            result["document"]
        )

        print("-" * 60)


    print("\nRAG + LLM Explanation")
    print("=" * 60)


    explanation = generate_rag_explanation(
        query=query,
        retrieved_documents=retrieved_documents
    )


    print(explanation)