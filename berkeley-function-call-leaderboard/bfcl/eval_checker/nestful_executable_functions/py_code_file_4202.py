import torch



def double_channels(tensor: torch.Tensor) -> torch.Tensor:

    """

    Takes a tensor of shape (N, H, C) and returns a tensor of shape (N, H, C * 2) where the new tensor is the original tensor concatenated with itself.

    Args:

        tensor: The input tensor.

    """

    return torch.cat([tensor, tensor], dim=2)

