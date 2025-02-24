from typing import List



def rotate_matrix_90(matrix: List[List[int]]) -> List[List[int]]:

    """Rotates a square matrix counter-clockwise by 90 degrees.



    Args:

        matrix: The input square matrix.



    Returns:

        The rotated matrix.

    """

    transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    rotated = [row[::-1] for row in transposed]



    return rotated

