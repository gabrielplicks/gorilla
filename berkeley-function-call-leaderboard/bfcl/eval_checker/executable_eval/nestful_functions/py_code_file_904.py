import numpy as np



def concatenate_arrays(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    """Concatenates two numpy 2D arrays along the second axis and returns a numpy 3D array.

    Args:

        x: A numpy 2D array of shape (N, M).

        y: A numpy 2D array of shape (N, K).

    """

    result = np.concatenate((x, y), axis=1)

    return np.expand_dims(result, axis=0)

