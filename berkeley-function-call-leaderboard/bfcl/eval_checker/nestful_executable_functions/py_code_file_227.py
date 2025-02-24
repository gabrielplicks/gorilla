from typing import List



def generate_symmetric_matrix(n: int) -> List[List[int]]:

    """Generates an n x n matrix filled with 0s and 1s that is symmetric along the main diagonal.

    The top-left entry should be 1, and its diagonal opposite should be 0.



    Args:

        n: The size of the matrix.



    Returns:

        A symmetric matrix of size n x n.

    """

    matrix = [[1 if i == j else 0 for i in range(n)] for j in range(n)]

    return matrix

