from typing import List



def rotate_cells_clockwise(matrix: List[List[int]]) -> List[List[int]]:

    """Rotates the given matrix clockwise by 90 degrees.



    The matrix is represented as a 2D array of integers. The function returns a new matrix rotated by 90 degrees clockwise.



    Args:

        matrix: The input matrix represented as a 2D array of integers.



    Returns:

        A new matrix rotated by 90 degrees clockwise.

    """

    n = len(matrix)

    rotated_matrix = [[0] * n for _ in range(n)]

    for i in range(n):

        for j in range(n):

            rotated_matrix[j][i] = matrix[i][j]

    for i in range(n):

        rotated_matrix[i] = rotated_matrix[i][::-1]

    return rotated_matrix

