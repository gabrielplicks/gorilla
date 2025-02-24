from typing import List



def create_identity_matrix(size: int) -> List[List[int]]:

    """Creates a square matrix of the given size, where each element on the main diagonal is 1, and all other elements are 0.



    Args:

        size: The size of the matrix.



    Returns:

        A list of lists, where each sublist represents a row of the matrix.

    """

    matrix = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):

        for j in range(size):

            if i == j:

                matrix[i][j] = 1

    return matrix

