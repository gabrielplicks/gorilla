import numpy as np



def l2_distance(a: np.ndarray, b: np.ndarray) -> float:

    """Calculates the L2 distance between two numpy arrays or lists `a` and `b`.



    Args:

        a: A numpy array or list of float numbers.

        b: A numpy array or list of float numbers with the same shape as `a`.



    Returns:

        A single float value representing the L2 distance between the two arrays.

    """

    squared_difference = np.sum((a - b) ** 2)

    return np.sqrt(squared_difference)

