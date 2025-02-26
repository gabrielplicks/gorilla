from typing import List



def initialize_list(rows: int, cols: int, val: int) -> List[List[int]]:

    """Initializes a two-dimensional list with the given dimensions and a default value.



    Args:

        rows: The number of rows in the list.

        cols: The number of columns in the list.

        val: The default value for each element in the list.

    """

    matrix = [[val] * cols for _ in range(rows)]

    return matrix

