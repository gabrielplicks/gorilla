import numpy as np



def vectorized_variance(data: np.ndarray) -> float:

    """Calculates the variance of a given array of numbers using vectorization and numerical operations.

    Args:

        data: A numpy array of numbers.

    """

    mean = np.mean(data)

    diff_squared = (data - mean) ** 2

    sum_diff_squared = np.sum(diff_squared)

    variance = sum_diff_squared / (len(data) - 1)

    return variance

