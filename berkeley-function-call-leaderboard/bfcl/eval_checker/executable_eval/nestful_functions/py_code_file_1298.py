import random

random.seed(42)
from typing import List



def generate_sql_statement(strings: List[str], table_name: str) -> str:

    """Generates a SQL statement that inserts all strings into a table with a dynamically generated name.



    Args:

        strings: A list of strings to be inserted into the table.

        table_name: The base name of the table.



    Returns:

        A SQL statement that inserts all strings into the table.

    """

    random_suffix = str(random.randint(10000, 99999))

    unique_table_name = f"{table_name}_{random_suffix}"

    sql_statement = "INSERT INTO"

    sql_statement += f" {unique_table_name}"

    sql_statement += " (string) VALUES"

    sql_statement += ", ".join(f"('{string}')" for string in strings)

    sql_statement += ";"



    return sql_statement

