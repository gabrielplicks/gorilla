import sqlite3



def execute_sql(sql: str, db: sqlite3.Connection) -> list[tuple]:

    """Executes an SQL query against a database and returns the results as a list of tuples.

    Handles any exceptions and returns an empty list when a sqlite3.Error is encountered.

    Args:

        sql: A string representing an SQL query.

        db: A database connection object.

    """

    try:

        cursor = db.execute(sql)

        results = cursor.fetchall()

        return results

    except sqlite3.Error:

        return []

