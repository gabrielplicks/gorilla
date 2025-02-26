from typing import Any



def count_rows_in_table(db_conn: Any, table: str) -> int:

    """Counts the number of rows in a table in a database.



    Args:

        db_conn: A database connection object.

        table: The name of the table to count rows in.



    Returns:

        The number of rows in the table.

    """

    query = f"SELECT COUNT(*) FROM {table}"

    result = db_conn.execute(query)

    count = result.fetchone()[0]

    return count

