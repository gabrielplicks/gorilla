import torch



def unsqueeze_tensor(tensor: torch.Tensor, dim: int) -> torch.Tensor:

    """Performs a generic unsqueeze operation on a PyTorch tensor.

    Args:

        tensor: The input PyTorch tensor.

        dim: The dimension index at which to insert a new dimension of size 1.

    """

    return tensor.unsqueeze(dim)

