import torch



def tensor_log(x: torch.Tensor) -> torch.Tensor:

    """Calculates the logarithm of a tensor.



    Args:

        x: The input tensor.



    Raises:

        ValueError: If the input tensor contains any non-positive values.

    """

    if torch.any(x <= 0):

        raise ValueError("Invalid input: tensor must contain only positive values.")

    return x.log()

