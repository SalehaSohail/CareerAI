"""
This file includes the functions required to extract job
information from the PostgreSQL database. In the function called
get_all_jobs(), a database connection is first established, and then
a SQL query is executed in order to fetch all the records of
available jobs. In the query, only those columns are extracted that
are currently required by CareerAI, including job title, company name,
location, employment type, experience level, and description of the
job. The fetched records are stored in the form of a list and can be
further utilized in the application for job matching and ranking.
After extracting the information, the cursor and database connection
are closed in order to free the resources. All of this code is placed
inside a function because as more and more features are added to the
application, it will help make the code reusable.
"""

from database import get_connection
def get_all_jobs():
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
            description
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
        }

        jobs.append(job)

    return jobs


if __name__ == "__main__":
    jobs = get_all_jobs()

    print(jobs)