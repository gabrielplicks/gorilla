import numpy as np



def create_matrix_from_vector(vector: np.ndarray) -> np.ndarray:

    """Creates a matrix from a vector using numpy array broadcasting and vectorization.



    The function takes an array of shape `(n,)` (i.e., a vector of length `n`) and returns an array of shape `(n, n)` (i.e., a matrix with `n` rows and `n` columns) where the `i`-th row is `[x[i]] * n`.



    Args:

        vector: The input vector.



    Returns:

        The output matrix.

    """

    vector = vector.reshape((-1, 1))

    return vector * np.ones((1, vector.shape[0]))

