from typing import List, Tuple



def zeros_matrix(m: int, n: int) -> List[List[int]]:

    """Creates a m x n matrix filled with zeros.



    Args:

        m: The number of rows in the matrix.

        n: The number of columns in the matrix.

    """

    matrix = [[0 for _ in range(n)] for _ in range(m)]

    return matrix

