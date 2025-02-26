import numpy as np



def min_max_normalize(tensor: np.ndarray) -> np.ndarray:

    """Performs min-max normalization on a tensor of shape (N, 3, H, W).



    Args:

        tensor: The input tensor.



    Returns:

        The normalized tensor.

    """

    min_val = np.min(tensor, axis=(1, 2, 3), keepdims=True)

    max_val = np.max(tensor, axis=(1, 2, 3), keepdims=True)

    norm_tensor = (tensor - min_val) / (max_val - min_val)

    return norm_tensor

