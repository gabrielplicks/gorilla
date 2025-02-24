from typing import List



def matrix_multiply_scalar(matrix: List[List[float]], scalar: float) -> List[List[float]]:

    """Multiplies a matrix by a scalar without modifying the input matrix.



    Args:

        matrix: The input matrix.

        scalar: The scalar to multiply the matrix by.



    Returns:

        The resulting matrix.

    """

    rows = len(matrix)

    cols = len(matrix[0])

    result = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):

        for j in range(cols):

            result[i][j] = matrix[i][j] * scalar

    return result

