
"""
This file contains the skill-matching logic for CareerAI.

The purpose of this module is to compare the skills required by a
job with the skills found in a user's resume.

The module identifies matched skills, missing skills, and calculates
a skill-match percentage. This information will later be combined
with the TF-IDF similarity score and other AI-based matching methods.
"""

import re


def normalize_skill(skill):
    """
    Normalize a skill before comparing it.

    This converts the skill to lowercase, removes unnecessary
    characters, and normalizes extra spaces.
    """

    skill = skill.lower().strip()

    skill = re.sub(r"[^a-z0-9+#.\-/ ]", "", skill)

    skill = re.sub(r"\s+", " ", skill)

    return skill


def extract_skills_from_resume(resume_text, skills_list):
    """
    Find which required skills appear in the resume text.
    """

    normalized_resume = normalize_skill(resume_text)

    matched_skills = []

    for skill in skills_list:
        normalized_skill = normalize_skill(skill)

        if normalized_skill in normalized_resume:
            matched_skills.append(skill)

    return matched_skills


def calculate_skill_match(resume_text, required_skills):
    """
    Compare the resume with a job's required skills.

    Returns matched skills, missing skills, and the
    overall skill-match percentage.
    """

    if not required_skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "skill_match_percentage": 0.0
        }

    job_skills = [
        skill.strip()
        for skill in required_skills.split(",")
        if skill.strip()
    ]

    matched_skills = extract_skills_from_resume(
        resume_text,
        job_skills
    )

    normalized_matched_skills = {
        normalize_skill(skill)
        for skill in matched_skills
    }

    missing_skills = []

    for skill in job_skills:
        if normalize_skill(skill) not in normalized_matched_skills:
            missing_skills.append(skill)

    skill_match_percentage = (
        len(matched_skills) / len(job_skills)
    ) * 100

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_percentage": round(
            skill_match_percentage,
            2
        )
    }

