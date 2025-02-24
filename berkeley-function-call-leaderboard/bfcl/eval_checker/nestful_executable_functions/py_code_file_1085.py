import numpy as np



def sum_along_dims(A: np.ndarray, indices: List[int]) -> np.ndarray:

    """

    Sums a numpy array `A` along the specified dimensions.



    Args:

        A: The numpy array to be summed.

        indices: A list of integers representing the dimensions along which `A` should be summed.

    """

    dims = tuple(indices)

    result = np.sum(A, axis=dims)

    result = result.astype(A.dtype)

    return result

