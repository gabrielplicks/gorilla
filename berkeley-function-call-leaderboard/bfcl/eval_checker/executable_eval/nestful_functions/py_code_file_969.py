import sqlite3

from typing import Tuple



def insert_file_into_database(conn: sqlite3.Connection, file_path: str) -> int:

    """Inserts the content of a file into a SQLite database.



    Args:

        conn: The SQLite database connection.

        file_path: The path to the file to be inserted.



    Returns:

        The number of rows inserted into the database.

    """

    with open(file_path, 'r') as file:

        content = file.read()

    conn.execute(

        "CREATE TABLE IF NOT EXISTS file_data (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, length INTEGER)"

    )

    conn.execute(

        "INSERT INTO file_data (content, length) VALUES (?, ?)",

        (content, len(content)),

    )

    inserted_rows = conn.execute("SELECT changes()").fetchone()[0]



    return inserted_rows

