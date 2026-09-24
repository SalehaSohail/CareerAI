"""
RAG document preparation module for CareerAI.

This module converts job information from PostgreSQL
into documents that can be used by the RAG system.
"""

from jobs import get_all_jobs


def create_job_documents():
    """
    Retrieve jobs from PostgreSQL and convert them
    into text documents for RAG retrieval.
    """

    jobs = get_all_jobs()

    documents = []

    for job in jobs:

        document = f"""
Job Title: {job["title"]}

Company: {job["company"]}

Location: {job["location"]}

Employment Type: {job["employment_type"]}

Experience Level: {job["experience_level"]}

Required Skills:
{job["required_skills"]}

Job Description:
{job["description"]}
"""

        documents.append(
            document.strip()
        )

    return documents


if __name__ == "__main__":

    print("RAG DOCUMENT FILE STARTED")

    documents = create_job_documents()

    print(
        f"\nTotal RAG documents: {len(documents)}"
    )

    print("\nFirst document:")
    print("=" * 60)
    print(documents[0])