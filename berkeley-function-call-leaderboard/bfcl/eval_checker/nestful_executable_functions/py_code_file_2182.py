import numpy as np

import torch

from typing import Union



def numpy_to_pytorch(arr: np.ndarray) -> torch.Tensor:

    """Converts a Numpy array to a PyTorch tensor.



    Args:

        arr: The input Numpy array.



    Returns:

        A PyTorch tensor with the same dimensions as the input array.

    """

    dims = arr.shape

    if len(dims) > 1:

        return torch.from_numpy(arr)

    else:

        return torch.tensor(arr)

