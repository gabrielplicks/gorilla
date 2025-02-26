import torch

from typing import Tuple



def sample_tensor(tensor: torch.Tensor, Q: int) -> torch.Tensor:

    """

    Samples Q columns from a 2D tensor without replacement.



    Args:

        tensor: The input 2D tensor.

        Q: The number of columns to sample.



    Returns:

        A 2D tensor of shape (N, Q), where N is the number of rows in the original tensor.

    """

    if len(tensor.shape) != 2:

        raise ValueError("Input tensor must be a 2D tensor")

    N, P = tensor.shape

    permutation = torch.randperm(P)

    sampled_indices = permutation[:Q]

    sampled_columns = tensor[:, sampled_indices]



    return sampled_columns

