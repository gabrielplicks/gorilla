from typing import List, Tuple



def skew_symmetric_matrix(vector: List[float]) -> List[List[float]]:

    """Creates a 3x3 skew-symmetric matrix from a 3D vector.



    A skew-symmetric matrix is a square matrix whose transpose equals its negative.



    Args:

        vector: A 3D vector represented as a list or tuple of 3 elements.



    Returns:

        A 3x3 skew-symmetric matrix represented as a list of lists.

    """

    matrix = [[0] * 3 for _ in range(3)]



    matrix[0][1] = -vector[2]

    matrix[0][2] = vector[1]

    matrix[1][0] = vector[2]

    matrix[1][2] = -vector[0]

    matrix[2][0] = -vector[1]

    matrix[2][1] = vector[0]



    return matrix

