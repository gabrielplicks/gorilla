import sqlite3

from typing import List, Tuple



def get_users_by_date(conn: sqlite3.Connection, start_date: str, end_date: str) -> List[Tuple]:

    """

    Queries the database for all users who joined within the specified date range.

    The function takes three arguments: the database connection object, the start date, and the end date.

    The function returns a list of tuples, with each tuple representing the user information (id, username, email, and date of joining).

    Args:

        conn: The database connection object.

        start_date: The start date of the range.

        end_date: The end date of the range.

    """

    with conn:

        cursor = conn.cursor()

        cursor.execute(f'SELECT * FROM users WHERE date_joined BETWEEN "{start_date}" AND "{end_date}" ORDER BY id')

        return cursor.fetchall()

