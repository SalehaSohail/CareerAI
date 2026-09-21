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
                "AI explanation is temporarily unavailable. "
                "Please try the analysis again."
            )

        if content.strip().lower() == "user safety: safe":
            return (
                "AI explanation is temporarily unavailable. "
                "Please try the analysis again."
            )

        return content

    except Exception as error:

        print(
            "LLM Error:",
            error
        )

        return (
            "AI explanation is temporarily unavailable. "
            "The job matching results are still available."
        )