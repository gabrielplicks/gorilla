import numpy as np



def scale_and_shift_numpy_array(array: np.ndarray, alpha: float = 1, beta: float = 0) -> np.ndarray:

    """Transforms an Numpy array by element-wise scaling and shifting.



    Args:

        array: The input Numpy array.

        alpha: The scaling parameter.

        beta: The shifting parameter.



    Returns:

        The scaled and shifted Numpy array.

    """

    array = np.array(array)

    scaled_and_shifted_array = np.empty_like(array)

    for i in range(array.size):

        scaled_value = alpha * array.flat[i] + beta

        scaled_and_shifted_array.flat[i] = scaled_value



    return scaled_and_shifted_array

