import sqlite3



def create_sqlite_connection(db_name: str) -> sqlite3.Connection:

    """Creates a SQLite in-memory database connection and returns it.

    If the database name already exists, it returns the existing database connection.

    If the database name does not exist, it creates a new database, connects to it, and returns the connection object.

    Args:

        db_name: The name of the database.

    """

    conn = None

    try:

        conn = sqlite3.connect(db_name)

    except sqlite3.OperationalError:

        conn = sqlite3.connect(db_name)



    return conn

