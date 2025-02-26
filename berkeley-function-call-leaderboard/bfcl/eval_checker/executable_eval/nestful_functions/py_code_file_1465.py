import torch

from typing import Union



def zeros_like(tensor: torch.Tensor, dtype: Union[torch.dtype, None] = None) -> torch.Tensor:

    """Creates a tensor of zeros with the same shape as a given tensor.

    The function supports broadcasting operations and creates the zeros tensor in the same data type as the input tensor.

    It also allows for indexing the input tensor using Numpy-like indexing.



    Args:

        tensor: The input tensor.

        dtype: The data type of the output tensor. If not provided, the data type of the input tensor is used.

    """

    shape = tensor.shape

    dtype = dtype or tensor.dtype

    zeros = torch.zeros(*shape, dtype=dtype)

    return zeros[...]

