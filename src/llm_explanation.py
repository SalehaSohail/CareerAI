

"""
LLM explanation module for CareerAI.

The LLM receives the matching results calculated by CareerAI
and explains the job match in simple language.

The LLM does NOT calculate the matching score.
It only explains the existing results.
"""

import os

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def generate_job_explanation(
    job_title,
    company,
    skill_match,
    matched_skills,
    missing_skills,
    tfidf_score,
    semantic_score
):
    """
    Generate a grounded natural-language explanation
    for a resume-job match.
    """

    prompt = f"""
You are an AI career assistant inside a job-matching application.

Your task is to explain the matching results provided by the
CareerAI matching system.

IMPORTANT RULES:

- Use ONLY the information provided below.
- Do not invent skills, experience, qualifications, or achievements.
- Do not create new percentages or scores.
- Do not predict whether the candidate will get the job.
- Do not say the candidate is guaranteed to be a good fit.
- The Skill Match percentage is calculated by CareerAI and must be
  reported exactly if mentioned.
- TF-IDF and Semantic Similarity are technical similarity signals,
  not hiring probabilities.

Job Title:
{job_title}

Company:
{company}

Skill Match:
{skill_match}%

Matched Skills:
{", ".join(matched_skills)}

Missing Skills:
{", ".join(missing_skills)}

TF-IDF Similarity:
{tfidf_score:.4f}

Semantic Similarity:
{semantic_score:.4f}


Provide the explanation using exactly these sections:

1. Match Summary
Briefly explain how the candidate's provided skills relate to the job.

2. Matched Skills
Explain the most relevant skills already present.

3. Missing Skills
Explain the important skills that are currently missing.

4. Improvement Suggestions
Give practical suggestions for improving the missing skills.

Do not introduce any additional numerical percentages or predictions.

Keep the explanation concise and practical.
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    from jobs import get_all_jobs

    from resume import (
        extract_text_from_pdf,
        clean_resume_text
    )

    from matching import calculate_job_matches


    # -----------------------------
    # Load resume
    # -----------------------------

    resume_path = r"D:\CareerAI_Resume\Saleha CV.pdf"

    raw_resume_text = extract_text_from_pdf(
        resume_path
    )

    resume_text = clean_resume_text(
        raw_resume_text
    )


    # -----------------------------
    # Load jobs
    # -----------------------------

    jobs = get_all_jobs()


    # -----------------------------
    # Calculate job matches
    # -----------------------------

    matches = calculate_job_matches(
        resume_text,
        jobs
    )


    # -----------------------------
    # Select best matching job
    # -----------------------------

    best_match = matches[0]

    job = best_match["job"]


    # -----------------------------
    # Generate LLM explanation
    # -----------------------------

    explanation = generate_job_explanation(
        job_title=job["title"],
        company=job["company"],
        skill_match=best_match[
            "skill_match_percentage"
        ],
        matched_skills=best_match[
            "matched_skills"
        ],
        missing_skills=best_match[
            "missing_skills"
        ],
        tfidf_score=best_match[
            "tfidf_score"
        ],
        semantic_score=best_match[
            "semantic_score"
        ]
    )


    # -----------------------------
    # Display result
    # -----------------------------

    print("\nCareerAI LLM Explanation")
    print("=" * 50)

    print(
        f"\nJob: {job['title']}"
    )

    print(
        f"Company: {job['company']}"
    )

    print(
        f"Skill Match: "
        f"{best_match['skill_match_percentage']:.2f}%"
    )

    print(
        f"TF-IDF Similarity: "
        f"{best_match['tfidf_score']:.4f}"
    )

    print(
        f"Semantic Similarity: "
        f"{best_match['semantic_score']:.4f}"
    )

    print("\nExplanation:")
    print(explanation)