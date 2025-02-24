import torch



def identity_matrix(tensor: torch.Tensor) -> torch.Tensor:

    """Returns the identity matrix of a given tensor.



    Args:

        tensor: The input tensor.

    """

    identity = torch.zeros_like(tensor)

    identity.fill_diagonal_(1)

    return identity

