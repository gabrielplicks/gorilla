from typing import Any



def insert_into_database(value: Any, connection: Any) -> int:

    """Inserts the given value into a database and returns the newly created row ID.



    Args:

        value: The value to be inserted into the database.

        connection: The database connection object.

    """

    with connection:

        cursor = connection.cursor()

        cursor.execute("INSERT INTO my_table (value) VALUES (?)", (value,))

        return cursor.lastrowid

