from typing import List

import sqlite3



def table_column_names(conn: sqlite3.Connection, table_name: str) -> List[str]:

    """

    Returns a list containing the names of the columns in the specified table.



    Args:

        conn: The database connection.

        table_name: The name of the table.

    """

    cursor = conn.cursor()

    cursor.execute(f'SELECT * FROM {table_name} LIMIT 1')

    column_names = [description[0] for description in cursor.description]

    cursor.close()



    return column_names

