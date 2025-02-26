import numpy as np



def one_dim_array(arr: np.ndarray) -> np.ndarray:

    """Removes all but one dimension from a N-dimensional array.



    Args:

        arr: The input array.



    Returns:

        A 1-dimensional array with the same values, but without the innermost dimension.

    """

    return arr.ravel()

