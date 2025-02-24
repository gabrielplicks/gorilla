import numpy as np



def scale_tensor(data: np.ndarray) -> np.ndarray:

    """Scales a 3-dimensional tensor of real-valued data so that the mean of the tensor is 0 and the standard deviation is 1.



    Args:

        data: The input tensor as a NumPy array.



    Returns:

        A new NumPy array containing the scaled tensor.

    """

    mean = data.mean()

    std = data.std()

    centered_data = data - mean

    scaled_data = centered_data / std

    return np.array(scaled_data)

