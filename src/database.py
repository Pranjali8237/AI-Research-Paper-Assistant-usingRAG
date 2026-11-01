import sqlite3
from pathlib import Path
from datetime import datetime


# Database location
DATABASE_PATH = Path("data/research_assistant.db")


def get_connection():

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return sqlite3.connect(
        DATABASE_PATH
    )


# ------------------------------------------------
# CREATE TABLES
# ------------------------------------------------

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    # Research papers table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            filename TEXT UNIQUE NOT NULL,

            uploaded_at TEXT NOT NULL
        )
        """
    )


    # Questions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS questions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            question TEXT NOT NULL,

            answer TEXT NOT NULL,

            sources TEXT,

            asked_at TEXT NOT NULL
        )
        """
    )


    connection.commit()

    connection.close()


# ------------------------------------------------
# PAPER FUNCTIONS
# ------------------------------------------------

def add_paper(filename):

    connection = get_connection()

    cursor = connection.cursor()


    uploaded_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    cursor.execute(
        """
        INSERT OR IGNORE INTO papers
        (filename, uploaded_at)

        VALUES (?, ?)
        """,
        (
            filename,
            uploaded_at
        )
    )


    connection.commit()

    connection.close()


def get_papers():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT id, filename, uploaded_at
        FROM papers
        ORDER BY uploaded_at DESC
        """
    )


    papers = cursor.fetchall()

    connection.close()

    return papers


def delete_paper(paper_id):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM papers
        WHERE id = ?
        """,
        (paper_id,)
    )


    connection.commit()

    connection.close()


def clear_papers():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        "DELETE FROM papers"
    )


    connection.commit()

    connection.close()


# ------------------------------------------------
# QUESTION FUNCTIONS
# ------------------------------------------------

def add_question(
    question,
    answer,
    sources
):

    connection = get_connection()

    cursor = connection.cursor()


    asked_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    source_text = ", ".join(
        sources
    )


    cursor.execute(
        """
        INSERT INTO questions
        (question, answer, sources, asked_at)

        VALUES (?, ?, ?, ?)
        """,
        (
            question,
            answer,
            source_text,
            asked_at
        )
    )


    connection.commit()

    connection.close()


def get_questions(
    search_text=""
):

    connection = get_connection()

    cursor = connection.cursor()


    if search_text.strip():

        cursor.execute(
            """
            SELECT id, question, answer,
                   sources, asked_at

            FROM questions

            WHERE question LIKE ?

            ORDER BY asked_at DESC
            """,
            (
                f"%{search_text}%",
            )
        )

    else:

        cursor.execute(
            """
            SELECT id, question, answer,
                   sources, asked_at

            FROM questions

            ORDER BY asked_at DESC
            """
        )


    questions = cursor.fetchall()

    connection.close()

    return questions


def delete_question(
    question_id
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM questions
        WHERE id = ?
        """,
        (question_id,)
    )


    connection.commit()

    connection.close()


def clear_questions():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        "DELETE FROM questions"
    )


    connection.commit()

    connection.close()