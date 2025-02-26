from typing import List



def product_of_matrices(matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:

    """Computes the product of two 3x3 matrices.



    The matrices are represented as lists of lists (each sublist represents a row).

    The function returns a new 3x3 matrix (represented as a list of lists) that is

    the product of the input matrices.



    Args:

        matrix1: The first 3x3 matrix.

        matrix2: The second 3x3 matrix.

    """

    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

    for i in range(len(matrix1)):

        for j in range(len(matrix2[0])):

            for k in range(len(matrix2)):

                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

