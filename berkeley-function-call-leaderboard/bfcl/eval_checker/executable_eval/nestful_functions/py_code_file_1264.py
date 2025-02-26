import torch



def minmax(tensor: torch.Tensor) -> torch.Tensor:

    """Calculates the minimum and maximum values along each column of a 2-dimensional PyTorch tensor.



    Args:

        tensor: A 2-dimensional PyTorch tensor containing floating-point numbers.



    Returns:

        A 1-dimensional PyTorch tensor containing the minimum and maximum values along each column.

    """

    if tensor.size(0) == 0:

        return torch.empty(0)



    return torch.stack((torch.min(tensor, dim=0)[0], torch.max(tensor, dim=0)[0]))

