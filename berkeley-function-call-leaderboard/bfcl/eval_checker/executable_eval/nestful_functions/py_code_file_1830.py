from typing import List



def generate_zeros_matrix(n: int, m: int) -> List[List[int]]:

    """Generates a two-dimensional list of zeros, where each sublist has `n` values.



    Args:

        n: The number of columns in the matrix.

        m: The number of rows in the matrix.



    Returns:

        A two-dimensional list of zeros with the specified dimensions.

    """

    matrix = []

    for i in range(m):

        row = [0] * n

        matrix.append(row)

    return matrix

