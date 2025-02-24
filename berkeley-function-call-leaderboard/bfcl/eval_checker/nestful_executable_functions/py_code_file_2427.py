from typing import List, Dict



def select_column(table: List[Dict], column_name: str) -> List:

    """Extracts the values from the column specified by the `column_name` parameter from a list of dictionaries.

    Args:

        table: A list of dictionaries representing rows in a database table.

        column_name: The name of the column to extract values from.

    """

    return [row.get(column_name, None) for row in table]

