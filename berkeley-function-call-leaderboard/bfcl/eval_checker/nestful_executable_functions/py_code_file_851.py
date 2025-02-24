import torch



def transpose_features(x: torch.Tensor) -> torch.Tensor:

    """Transposes a tensor of shape (B, N, 3) into a tensor of shape (B, 3, N).



    Args:

        x: The input tensor of shape (B, N, 3).



    Returns:

        The transposed tensor of shape (B, 3, N).

    """

    return x.permute(0, 2, 1)

