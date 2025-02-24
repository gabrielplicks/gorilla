import numpy as np



def maximum_value_diagonal(arr: np.ndarray, v: float) -> float:

    """

    Finds the maximum value of the array elements that are both greater than or equal to a given value `v`

    and located on or above the diagonal.



    Args:

        arr: A 2D Numpy array.

        v: The given value.

    """

    max_val = -np.inf

    for row in range(arr.shape[0]):

        for col in range(arr.shape[1]):

            if row <= col and arr[row, col] >= v:

                max_val = max(max_val, arr[row, col])

    return max_val

