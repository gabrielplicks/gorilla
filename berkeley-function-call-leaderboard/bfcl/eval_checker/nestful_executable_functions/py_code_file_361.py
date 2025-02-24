import numpy as np



def matrix_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    """Calculates the product of two matrices.



    Args:

        a: A numpy array of shape (N, M).

        b: A numpy array of shape (M, K).



    Returns:

        A numpy array of shape (N, K) representing the product of the two matrices.

    """

    return np.matmul(a, b)

