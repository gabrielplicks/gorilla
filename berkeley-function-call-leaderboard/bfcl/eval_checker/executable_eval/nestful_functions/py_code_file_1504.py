import numpy as np



def normalize_mean_variance(arr: np.ndarray) -> np.ndarray:

    """Normalizes the mean and variance of a set of data in an array.



    Args:

        arr: The input array.



    Returns:

        The normalized array.

    """

    assert arr.dtype == np.float32, 'Input array should have a float32 data type!'

    arr_mean = np.mean(arr)

    arr_std = np.std(arr)

    normalized_arr = (arr - arr_mean) / arr_std

    return normalized_arr

