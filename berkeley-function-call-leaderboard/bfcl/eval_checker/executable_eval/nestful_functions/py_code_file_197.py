from typing import List



def generate_sql_delete_query(num_ids: List[int], str_values: List[str], table_name: str) -> str:

    """Generates a parametrized SQL DELETE query from a list of numeric IDs and a list of string values.



    Args:

        num_ids: A list of numeric IDs.

        str_values: A list of string values.

        table_name: A string representing the name of the database table.

    """

    num_ids_str = ', '.join(str(id) for id in num_ids)

    str_values_str = ', '.join(f"'{value}'" for value in str_values)

    return f"DELETE FROM {table_name} WHERE id IN ({num_ids_str}) AND value IN ({str_values_str});"

