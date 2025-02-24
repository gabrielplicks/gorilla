import torch

from typing import Union



def element_wise_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    """

    Performs element-wise addition of two PyTorch tensors, handling broadcasting by automatically expanding the smaller tensor to match the shape of the larger tensor.



    Args:

        a: The first tensor.

        b: The second tensor.



    Returns:

        The result of the element-wise addition as a new tensor.

    """

    return torch.add(a, b)

