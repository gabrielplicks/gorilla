import numpy as np



def matrix_to_index(matrix: np.ndarray) -> np.ndarray:

    """Converts a given matrix to a 1-D array by raveling the matrix.



    Args:

        matrix: The input matrix.



    Raises:

        AssertionError: If the input matrix is not a 2-D array.

    """

    assert matrix.ndim == 2, "Input matrix must be a 2-D array"

    return np.ravel(matrix)

