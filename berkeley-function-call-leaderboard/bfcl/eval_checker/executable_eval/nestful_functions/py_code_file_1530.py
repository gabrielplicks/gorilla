import numpy as np



def matrix_power(A: np.ndarray, k: int) -> np.ndarray:

    """Calculates the product of a square matrix A and its power k.



    Args:

        A: The input square matrix.

        k: The power to which the matrix is raised.



    Returns:

        The product of the matrix A and its power k.

    """

    if k == 0:

        n = len(A)

        return np.identity(n)

    else:

        A_k = np.copy(A)

        for _ in range(k-1):

            A_k = np.dot(A_k, A)

        return A_k

