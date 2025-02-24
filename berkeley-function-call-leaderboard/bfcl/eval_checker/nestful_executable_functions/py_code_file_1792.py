from typing import List, Union



def lookup_symmetric_table(table: List[List[Union[str, int]]], key: Union[str, int], fallback_value: Union[str, int] = None) -> Union[str, int]:

    """

    Looks up a value in a symmetric table given a key.

    Args:

        table: A 2-D list representing a symmetric table.

        key: The key to look up in the table.

        fallback_value: The value to return if the key is not found in the table.

    """

    if not table or not key:

        raise ValueError("Input is invalid")

    for entry in table:

        if entry[0] == key:

            return entry[1]

    return fallback_value

