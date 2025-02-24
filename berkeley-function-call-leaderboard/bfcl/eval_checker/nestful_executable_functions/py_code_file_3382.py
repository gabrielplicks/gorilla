import numpy as np



def elementwise_multiply_and_sum(array1: np.ndarray, array2: np.ndarray) -> float:

    """Performs element-wise multiplication of two 2D arrays and returns the sum of all elements in the resulting array.



    Args:

        array1: The first 2D array.

        array2: The second 2D array.



    Raises:

        AssertionError: If the input arrays do not have the same shape.

    """

    if array1.shape != array2.shape:

        raise AssertionError("The input arrays must have the same shape.")



    result = np.multiply(array1, array2)

    return np.sum(result)

