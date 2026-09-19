"""
The project uses PostgreSQL to store and organize data about jobs,
and, in the future, other information needed by the application.
Instead of writing the same database connection configuration in
every Python file separately, we have created a reusable
get_connection() function that can be imported wherever a database
connection is needed. The database credentials are taken from the
.env file instead of being hard-coded into the source code. This
helps keep sensitive information, such as the database password,
confidential. Any part of the project that needs to communicate
with PostgreSQL can simply import and use the get_connection()
function. Keeping the database connection logic in one place makes
the project easier to maintain, understand, and expand in the future.
"""

import os

import psycopg

from dotenv import load_dotenv

load_dotenv()


def get_connection():

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

