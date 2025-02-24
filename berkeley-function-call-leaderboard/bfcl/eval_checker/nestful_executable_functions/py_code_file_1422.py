import torch

from typing import Tuple



def compute_min_value_indices(x: torch.Tensor, k: int) -> torch.Tensor:

    """Computes the indices of the k smallest values in the tensor x.



    Args:

        x: The input tensor.

        k: The number of smallest values to find.



    Returns:

        The indices of the k smallest values in the tensor x.

    """

    sorted_x, indices = torch.sort(x)

    return indices[:k]

