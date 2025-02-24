import numpy as np



def find_mean_index(arr: np.ndarray) -> int:

    """Finds the index of an element in a 1D numpy array that is equal to the mean value of the array.

    Args:

        arr: The input 1D numpy array.

    Returns:

        The index of the single element that is equal to the mean value of the array, or -1 if there is no such element.

    """

    mean_val = np.mean(arr)

    for i, val in enumerate(arr):

        if val == mean_val:

            return i

    return -1

