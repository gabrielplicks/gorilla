from typing import List



def square_matrix(n: int, fill_value: int = 0) -> List[List[int]]:

    """Creates a square matrix of 0 and 1, given an integer `n` as the size of the matrix.

    The matrix should have `n` rows and `n` columns.

    The function takes a second argument called `fill_value` that defaults to 0.

    The function returns the matrix as a list of lists.



    Args:

        n: The size of the matrix.

        fill_value: The value to fill the matrix with. Defaults to 0.



    Returns:

        A square matrix of size `n` filled with `fill_value`.

    """

    return [[fill_value] * n for _ in range(n)]

