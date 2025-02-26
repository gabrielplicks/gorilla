from typing import Union



def generate_insert_sql(id: int, name: str, email: str, phone: str) -> str:

    """Generates a SQL statement for inserting a new record into a user table.



    Args:

        id: The unique identifier for the user.

        name: The name of the user.

        email: The email address of the user.

        phone: The phone number of the user.



    Returns:

        A string that represents the SQL statement for inserting a new record into the user table.

    """

    sql = "INSERT INTO user (id, name, email, phone) VALUES ({}, '{}', '{}', '{}')"

    return sql.format(id, name, email, phone)

