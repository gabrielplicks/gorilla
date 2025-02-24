import math

import torch



def log1p_exp(x: torch.Tensor) -> torch.Tensor:

    """Calculates log(1 + exp(x)) using a numerically stable implementation.



    Args:

        x: The input tensor.



    Returns:

        The result of log(1 + exp(x)).

    """

    max_val = torch.max(x, torch.zeros_like(x))

    result = max_val + torch.log(1 + torch.exp(-torch.abs(x)))

    return result

