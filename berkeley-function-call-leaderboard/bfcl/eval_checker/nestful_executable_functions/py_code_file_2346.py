import numpy as np

import torch



def convert_array_to_tensor(array: np.ndarray) -> torch.Tensor:

    """Converts a given one-dimensional NumPy array into a PyTorch tensor.



    Args:

        array: The input one-dimensional NumPy array.



    Returns:

        A PyTorch tensor.

    """

    return torch.from_numpy(array)

