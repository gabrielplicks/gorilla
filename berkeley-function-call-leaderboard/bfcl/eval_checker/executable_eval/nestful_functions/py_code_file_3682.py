from typing import List



def rotate_matrix_90_cw(matrix: List[List[int]]) -> None:

    """Rotates a 2D matrix in-place by 90 degrees clockwise.



    Args:

        matrix: The input 2D matrix.

    """

    n = len(matrix)

    for i in range(n):

        for j in range(i, n):

            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]



    for i in range(n):

        for j in range(n // 2):

            matrix[i][j], matrix[i][n - j - 1] = matrix[i][n - j - 1], matrix[i][j]

