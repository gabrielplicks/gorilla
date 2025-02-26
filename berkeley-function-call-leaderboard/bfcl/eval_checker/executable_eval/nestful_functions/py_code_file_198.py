from typing import List



def create_zero_matrix(m: int, n: int) -> List[List[int]]:

    """Creates and returns an `m` by `n` zero-filled 2D matrix.



    Args:

        m: The number of rows in the matrix.

        n: The number of columns in the matrix.



    Raises:

        ValueError: If `m` or `n` is not a positive integer.

    """

    if not isinstance(m, int) or not isinstance(n, int) or m <= 0 or n <= 0:

        raise ValueError("m and n must be positive integers")

    matrix = [[0 for _ in range(n)] for _ in range(m)]

    for i in range(m):

        for j in range(n):

            matrix[i][j] = 0



    return matrix

