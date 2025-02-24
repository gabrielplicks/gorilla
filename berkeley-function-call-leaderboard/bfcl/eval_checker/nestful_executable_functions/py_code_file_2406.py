from typing import List



def sum_matrix(matrix: List[List[int]]) -> int:

    """Calculates the sum of all elements in a two-dimensional square matrix.



    Args:

        matrix: A two-dimensional square matrix of integers.



    Returns:

        The sum of all elements in the matrix.

    """

    total = 0

    for row in matrix:

        for element in row:

            total += element



    return total

