from typing import List, Union



def find_column_index(table: List[List[Union[str, int, float]]], column_name: str) -> int:

    """Finds the index of a column with the given name in a table.

    Args:

        table: The table represented as a list of lists, where each sublist is a row, and the first row contains the column headers.

        column_name: The name of the column to find.

    """

    for i, header in enumerate(table[0]):

        if header == column_name:

            return i

    return -1

