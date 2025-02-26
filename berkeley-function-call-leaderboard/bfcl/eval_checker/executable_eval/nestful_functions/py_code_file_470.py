import torch

from typing import Tuple



def norm_of_dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    """Computes the norm of the dot product of two tensors.



    Args:

        x: The first tensor.

        y: The second tensor.



    Returns:

        The norm of the dot product of the two tensors.

    """

    return torch.sqrt(torch.dot(x, y))

