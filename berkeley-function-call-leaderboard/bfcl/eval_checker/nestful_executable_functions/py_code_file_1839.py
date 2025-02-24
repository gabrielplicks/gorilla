import numpy as np



def transpose_2D_array(array: np.ndarray) -> np.ndarray:

    """Transposes a 2D NumPy array by switching the row and column indices.



    The transposed array has the same data as the original, but its rows and columns

    are swapped. The function handles any-sized 2D arrays, and returns `None` if a

    non-2D array is passed to it. If the input array is empty, the function returns

    an empty array.



    Args:

        array: The input 2D NumPy array.



    Returns:

        The transposed 2D NumPy array, or `None` if the input array is not 2D.

    """

    if array.ndim != 2:

        return None

    return array.T

