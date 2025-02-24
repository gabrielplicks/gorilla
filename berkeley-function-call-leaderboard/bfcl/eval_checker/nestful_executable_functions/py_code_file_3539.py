from typing import List, Dict



def get_values_from_column(dicts: List[Dict], column_name: str) -> List:

    """

    Returns a list of all values in the specified column from the dictionaries.

    If a dictionary does not have a value for the specified column, then the function adds `None` in its place.

    Args:

        dicts: A list of dictionaries.

        column_name: The name of the column to extract values from.

    """

    assert isinstance(dicts, list), "Input must be a list of dictionaries"

    assert isinstance(column_name, str), "Column name must be a string"

    return [d.get(column_name) for d in dicts]

