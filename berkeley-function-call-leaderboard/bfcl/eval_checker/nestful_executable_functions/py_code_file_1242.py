import torch



def remove_middle_dim(x: torch.Tensor) -> torch.Tensor:

    """

    Removes the middle dimension from a 3D tensor `x` of shape `(N, 1, 4)`,

    resulting in a tensor of shape `(N, 4)`.



    Args:

        x: A 3D tensor of shape `(N, 1, 4)`.



    Returns:

        A tensor of shape `(N, 4)`.

    """

    return x.squeeze(axis=1)

