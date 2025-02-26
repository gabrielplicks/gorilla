import sqlite3



def create_connection_and_cursor(db_file: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:

    """Creates a SQLite database connection and a cursor, and returns them.

    Args:

        db_file: The path to the database file. If the file does not exist, it will be created.

    """

    conn = sqlite3.connect(db_file)

    cursor = conn.cursor()

    return conn, cursor

