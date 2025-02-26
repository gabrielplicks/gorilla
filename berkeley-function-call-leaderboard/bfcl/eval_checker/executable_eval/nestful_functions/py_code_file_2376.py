import numpy as np



def mean_along_axis(arr: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:

    """Calculates the mean value of a 2D array along the specified axis.



    Args:

        arr: The 2D array to calculate the mean of.

        axis: The axis to calculate the mean along. If not specified, the mean of the flattened array is calculated.

        keepdims: Whether the output array should maintain the input's dimensions or be flattened.

    """

    return np.mean(arr, axis=axis, keepdims=keepdims)

