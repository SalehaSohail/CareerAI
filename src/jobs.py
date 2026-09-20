
"""
This file contains the database access logic for CareerAI jobs.

The purpose of this module is to retrieve job information from the
PostgreSQL database in a clean and reusable way. Instead of writing
SQL queries throughout the project, CareerAI keeps the job retrieval
logic in one place.

Each job is returned as a Python dictionary containing the job ID,
title, company, location, employment type, experience level,
description, and required skills.

Keeping database access separate from the machine learning logic
makes the project easier to maintain. The matching module can focus
on calculating similarity and skill matches without needing to know
how the job data is stored in PostgreSQL.

As CareerAI grows, this module can also be extended with functions
for searching jobs, filtering by location or experience level,
retrieving individual jobs, and managing job records.
"""

from database import get_connection


def get_all_jobs():
    """
    Retrieve all jobs from the PostgreSQL database.

    The function returns the jobs as a list of dictionaries so that
    other parts of CareerAI can access job information using clear
    field names instead of numeric tuple indexes.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            title,
            company,
            location,
            employment_type,
            experience_level,
            description,
            required_skills
        FROM jobs
        ORDER BY id;
    """)

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    jobs = []

    for row in rows:
        job = {
            "id": row[0],
            "title": row[1],
            "company": row[2],
            "location": row[3],
            "employment_type": row[4],
            "experience_level": row[5],
            "description": row[6],
            "required_skills": row[7],
        }

        jobs.append(job)

    return jobs


if __name__ == "__main__":
    jobs = get_all_jobs()

    for job in jobs:
        print(job)

