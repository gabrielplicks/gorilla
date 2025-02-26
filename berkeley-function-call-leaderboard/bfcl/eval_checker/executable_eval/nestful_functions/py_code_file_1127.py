import numpy as np



def calculate_mean_of_array(array: np.ndarray) -> float:

    """Calculates the mean of all the values in a NumPy array.



    Args:

        array: A NumPy array with a shape of (N, M), where N and M are positive integers.

    """

    column_means = np.mean(array, axis=0)

    return np.mean(column_means)

