import torch



def half_l2_norm(tensor: torch.Tensor) -> torch.Tensor:

    """Calculates the half of the L2-norm of a 2-dimensional torch tensor.



    Args:

        tensor: The input 2-dimensional torch tensor.



    Returns:

        The half of the L2-norm of the input tensor.

    """

    l2_norm = torch.norm(tensor, p=2)

    return l2_norm / 2

