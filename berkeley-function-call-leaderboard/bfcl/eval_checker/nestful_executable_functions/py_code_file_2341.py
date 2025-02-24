from typing import List, Dict, Set



def consistency_check(data: List[Dict[str, str]]) -> bool:

    """

    Checks whether all dictionaries in the data list have the same keys. If there are any

    missing keys, the function raises a KeyError to indicate the inconsistent keys.

    Args:

        data: A list of dictionaries, where each dictionary represents a row of data with

            column names as keys.

    Returns:

        True if all dictionaries have the same keys, False otherwise.

    """

    column_names: Set[str] = set()



    for row in data:

        row_keys: Set[str] = set(row.keys())

        if len(column_names) == 0:

            column_names = row_keys

        elif row_keys != column_names:

            raise KeyError(f"Inconsistent keys: {row_keys - column_names}")



    return True

