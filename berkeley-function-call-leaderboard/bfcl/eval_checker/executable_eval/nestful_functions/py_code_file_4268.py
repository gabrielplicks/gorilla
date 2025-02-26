import torch

from typing import Tuple



def calculate_sparsity(tensor: torch.Tensor) -> float:

    """Calculates the sparsity of a PyTorch tensor as the ratio of zero elements to total elements.



    Args:

        tensor: The PyTorch tensor to calculate the sparsity of.



    Returns:

        The sparsity of the tensor.

    """

    total_elements = 1

    for dim in tensor.size():

        total_elements *= dim

    num_zeros = torch.sum(tensor == 0).item()

    sparsity = num_zeros / total_elements

    return sparsity

