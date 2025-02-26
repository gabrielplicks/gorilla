import numpy as np



def four_axis_mean(input_array: np.ndarray) -> np.ndarray:

    """Calculates the mean of a four-dimensional input array along the first two axes.

    Args:

        input_array: The input array to calculate the mean of.

    """

    assert input_array.ndim >= 4, "Input array must be at least 4-dimensional"

    mean_array = np.mean(input_array, axis=(0, 1))

    return mean_array

