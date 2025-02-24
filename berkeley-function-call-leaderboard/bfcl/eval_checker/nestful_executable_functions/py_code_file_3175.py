import numpy as np



def gradient_matrix_multiplication(X: np.ndarray, W: np.ndarray) -> np.ndarray:

    """Calculates the first-order gradient of a matrix multiplication operation.

    Args:

        X: A two-dimensional matrix.

        W: A two-dimensional matrix.

    """

    return np.matmul(np.eye(W.shape[0]), np.zeros(W.shape))

