import torch

import torch.nn.functional as F



def normalize_unit_vectors(tensor: torch.Tensor) -> torch.Tensor:

    """

    Normalizes a PyTorch tensor of unit vectors along the last dimension.

    Args:

        tensor: The tensor to be normalized.

    """

    return F.normalize(tensor, dim=-1)

