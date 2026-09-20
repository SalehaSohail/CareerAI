
"""
This file contains sample job data for CareerAI.

The purpose of this module is to populate the PostgreSQL jobs table
with realistic job descriptions that can be used during development
and testing.

Each job contains information about the position, company, location,
employment type, experience level, required skills, and job
responsibilities. The descriptions are intentionally more detailed
than simple one-line examples because the job-matching system uses
the text of these descriptions to compare jobs with a user's resume.

The current jobs are development data and are not copied from real
companies or real job advertisements. Later, CareerAI can replace
these seed records with jobs collected from appropriate APIs or
other legitimate data sources.

Keeping the seed data in a separate file makes it easy to reset or
expand the development database without mixing sample data with
the application's main database and matching logic.
"""
from database import get_connection


jobs = [
    (
        "Junior Machine Learning Engineer",
        "TechVision",
        "Islamabad",
        "Full-time",
        "Entry-level",
        """
        We are looking for a Junior Machine Learning Engineer to
        support the development and evaluation of machine learning
        models. The role involves data preprocessing, exploratory
        data analysis, feature engineering, model training and model
        evaluation. Candidates should have knowledge of Python,
        Pandas, NumPy, scikit-learn and SQL. Experience with
        classification, regression, model evaluation and data
        pipelines is helpful. Familiarity with Git and basic
        software development practices is preferred.
        """
    ),

    (
        "Junior Data Scientist",
        "DataWorks",
        "Remote",
        "Full-time",
        "Entry-level",
        """
        We are hiring a Junior Data Scientist to analyze structured
        datasets and develop data-driven solutions. Responsibilities
        include data cleaning, exploratory data analysis, feature
        engineering, statistical analysis and predictive modeling.
        Strong Python and SQL skills are required, along with
        experience using Pandas, NumPy and scikit-learn. Knowledge
        of regression, classification, clustering, visualization
        and machine learning evaluation metrics is beneficial.
        Candidates should be comfortable working with databases and
        communicating analytical findings.
        """
    ),

    (
        "AI Engineer",
        "NexaAI",
        "Islamabad",
        "Full-time",
        "Entry-level",
        """
        We are looking for an entry-level AI Engineer to help build
        artificial intelligence applications. The role involves
        developing Python-based AI services, working with machine
        learning models, integrating APIs and experimenting with
        large language models. Knowledge of Python, machine learning,
        NLP, REST APIs and LLM concepts is required. Familiarity with
        FastAPI, embeddings, prompt engineering and model evaluation
        is a plus. Candidates should understand basic software
        engineering and Git workflows.
        """
    ),

    (
        "Python Developer",
        "CodeLabs",
        "Rawalpindi",
        "Full-time",
        "Entry-level",
        """
        We are looking for a Junior Python Developer to build and
        maintain backend applications and REST APIs. Candidates
        should understand Python programming, object-oriented
        programming, SQL, PostgreSQL and API development. Experience
        with FastAPI or similar Python web frameworks is preferred.
        The role includes database integration, writing clean code,
        debugging applications, testing APIs and using Git for
        version control. Knowledge of basic data structures and
        software development practices is expected.
        """
    ),

    (
        "Machine Learning Intern",
        "AILabs",
        "Remote",
        "Internship",
        "Entry-level",
        """
        We are looking for a Machine Learning Intern to assist the
        team with data preparation and machine learning experiments.
        Responsibilities include collecting and cleaning data,
        exploratory data analysis, feature engineering, training
        machine learning models and evaluating model performance.
        Candidates should have basic knowledge of Python, Pandas,
        NumPy and scikit-learn. Familiarity with classification,
        regression, datasets and machine learning metrics is useful.
        This role is suitable for students and recent graduates
        interested in practical machine learning work.
        """
    ),

    (
        "Data Analyst",
        "InsightTech",
        "Islamabad",
        "Full-time",
        "Entry-level",
        """
        We are hiring a Junior Data Analyst to work with business
        and operational data. The role includes writing SQL queries,
        cleaning datasets, performing exploratory data analysis,
        creating reports and identifying useful patterns in data.
        Candidates should have strong SQL and Python fundamentals
        and experience with Pandas and data visualization. Knowledge
        of PostgreSQL, statistical analysis, data validation and
        dashboard development is helpful. The ability to explain
        analytical findings clearly is important.
        """
    ),

    (
        "Junior AI Developer",
        "FutureSoft",
        "Remote",
        "Full-time",
        "Entry-level",
        """
        We are looking for a Junior AI Developer to create
        AI-powered software applications. The position involves
        working with Python, APIs, machine learning models and
        large language models. Candidates should understand basic
        NLP concepts, prompt engineering, API integration and
        structured data processing. Familiarity with LLM APIs,
        embeddings, retrieval augmented generation, FastAPI and
        Git is a plus. The role also involves testing AI features
        and improving application quality.
        """
    ),

    (
        "NLP Engineer",
        "LanguageAI",
        "Lahore",
        "Full-time",
        "Entry-level",
        """
        We are looking for a Junior NLP Engineer to work on natural
        language processing applications. Responsibilities include
        text preprocessing, feature extraction, machine learning
        experiments and evaluation of NLP models. Candidates should
        have knowledge of Python, machine learning, NLP and text
        processing. Experience with scikit-learn, PyTorch, tokenization,
        embeddings and language models is beneficial. Familiarity
        with speech or text-based AI systems is also useful.
        """
    ),

    (
        "Backend Python Developer",
        "CloudTech",
        "Remote",
        "Full-time",
        "Entry-level",
        """
        We are hiring a Junior Backend Python Developer to build
        reliable backend services and REST APIs. Candidates should
        understand Python, FastAPI, PostgreSQL, SQL and API
        development. Responsibilities include implementing backend
        features, designing database interactions, handling API
        requests, debugging and writing maintainable code.
        Experience with authentication, HTTP status codes, testing,
        Git and Docker is helpful. Knowledge of integrating external
        APIs and building scalable backend services is preferred.
        """
    ),

    (
        "Junior AI/ML Engineer",
        "InnovateAI",
        "Islamabad",
        "Full-time",
        "Entry-level",
        """
        We are looking for a Junior AI/ML Engineer to contribute to
        machine learning and artificial intelligence projects.
        Responsibilities include preparing datasets, performing
        exploratory data analysis, engineering features, training
        machine learning models and evaluating results. Candidates
        should have practical knowledge of Python, SQL, Pandas,
        NumPy and scikit-learn. Familiarity with PyTorch, APIs,
        NLP, deep learning and model deployment is a plus.
        Experience with Git and basic data pipeline development
        would be beneficial.
        """
    )
]


def seed_jobs():
    """
    Insert the development job records into the PostgreSQL database.

    The function uses executemany() so that all job records can be
    inserted efficiently using one SQL statement.
    """

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


if __name__ == "__main__":
    seed_jobs()






