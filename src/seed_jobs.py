"""
This file inserts initial job data into the CareerAI PostgreSQL database.
For the present phase of our work, we use a small set of realistic sample
jobs to create and test the database and job matching capability without
having an external job data source connected to our application yet.
Each job has various attributes like title, company name, location,
employment type, experience level, and job description. We have imported
the get_connection() function from our database module to re-use the
same connection code to our database in all other modules in this
project. Using the executemany() function, we can insert multiple jobs
using a single SQL command, and using the commit() command, the changes
are made permanent in PostgreSQL. The connection and cursor are closed
properly after inserting the records. This file is for testing purposes
only and can be removed later on when CareerAI begins getting jobs from
an external source.
"""


from database import get_connection


jobs = [
    (
        "Junior Machine Learning Engineer",
        "TechVision",
        "Islamabad",
        "Full-time",
        "Entry-level",
        "Work on machine learning models using Python, pandas, scikit-learn and SQL."
    ),
    (
        "Junior Data Scientist",
        "DataWorks",
        "Remote",
        "Full-time",
        "Entry-level",
        "Analyze datasets, build predictive models and create data-driven insights using Python and SQL."
    ),
    (
        "AI Engineer",
        "NexaAI",
        "Islamabad",
        "Full-time",
        "Entry-level",
        "Develop AI applications using Python, machine learning APIs and large language models."
    ),
    (
        "Python Developer",
        "CodeLabs",
        "Rawalpindi",
        "Full-time",
        "Entry-level",
        "Develop backend applications using Python, APIs, PostgreSQL and FastAPI."
    ),
    (
        "Machine Learning Intern",
        "AI Labs",
        "Remote",
        "Internship",
        "Entry-level",
        "Assist with machine learning experiments, data preprocessing and model evaluation."
    ),
    (
        "Data Analyst",
        "InsightTech",
        "Islamabad",
        "Full-time",
        "Entry-level",
        "Analyze business data using SQL, Python, pandas and visualization tools."
    ),
    (
        "Junior AI Developer",
        "FutureSoft",
        "Remote",
        "Full-time",
        "Entry-level",
        "Build AI-powered applications using Python, APIs, LLMs and prompt engineering."
    ),
    (
        "NLP Engineer",
        "LanguageAI",
        "Lahore",
        "Full-time",
        "Entry-level",
        "Work on natural language processing projects using Python, machine learning and text data."
    ),
    (
        "Backend Python Developer",
        "CloudTech",
        "Remote",
        "Full-time",
        "Entry-level",
        "Build REST APIs and backend services using Python, FastAPI and PostgreSQL."
    ),
    (
        "Junior AI/ML Engineer",
        "InnovateAI",
        "Islamabad",
        "Full-time",
        "Entry-level",
        "Develop machine learning and AI solutions using Python, SQL, scikit-learn and APIs."
    )
]


connection = get_connection()
cursor = connection.cursor()

cursor.executemany(
    """
    INSERT INTO jobs (
        title,
        company,
        location,
        employment_type,
        experience_level,
        description
    )
    VALUES (%s, %s, %s, %s, %s, %s);
    """,
    jobs
)

connection.commit()

cursor.close()
connection.close()

print(f"{len(jobs)} jobs inserted successfully!")