import torch

from typing import List



def get_data_by_indices(data: torch.Tensor, idx: List[int], target_idx: List[int]) -> torch.Tensor:

    """

    Extracts specific values from a PyTorch tensor based on a given set of indices.



    Args:

        data: The input tensor.

        idx: The list of indices.

        target_idx: The list of target indices.

    """

    target_values = torch.zeros(len(target_idx))

    for i, j in enumerate(target_idx):

        target_values[i] = data[idx[j], j]

    return target_values

