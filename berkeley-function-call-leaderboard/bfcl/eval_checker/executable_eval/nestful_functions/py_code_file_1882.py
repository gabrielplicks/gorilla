from typing import Tuple



def get_row_count(cursor: Tuple, query: str) -> int:

    """Executes the query against the cursor and fetches the number of rows from the cursor object.



    Args:

        cursor: The cursor object to execute the query against.

        query: The query string to execute.

    """

    cursor.execute(query)

    rows = cursor.fetchall()

    return len(rows)

