from typing import List, Dict, Union



def create_table_with_columns(columns: List[str], column_types: Dict[str, Union[type, None]] = None) -> List[List[Union[str, type, None]]]:

    """Creates a table with the specified columns and column types.



    Args:

        columns: A list of column names.

        column_types: A dictionary of column types. If not provided, all columns will be of type None.



    Returns:

        A list of lists representing the table.

    """

    table = []

    if column_types:

        table.append([column_types.get(col, None) for col in columns])

    table.append(columns)

    return table

