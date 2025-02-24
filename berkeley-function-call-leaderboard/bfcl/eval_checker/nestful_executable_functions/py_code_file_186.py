import torch

from typing import Tuple



def convert_to_sequence(tensor: torch.Tensor) -> torch.Tensor:

    """Converts a PyTorch tensor to a sequence of tensors with the same rank but varying along the first dimension.

    Concatenates all the tensors into a single tensor with an added batch dimension.



    Args:

        tensor: The input PyTorch tensor.



    Returns:

        The concatenated tensor with an added batch dimension.

    """

    chunks = torch.chunk(tensor, tensor.size(0))

    tensors = [torch.unsqueeze(chunk, 0) for chunk in chunks]

    return torch.cat(tensors, dim=0)

