import torch

from typing import Tuple



def tensor_logic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    """Creates a new Tensor that has the values of 1 where either `x` or `y` is 1, and 0 elsewhere.



    Args:

        x: A Torch Tensor.

        y: A Torch Tensor of the same size as `x`.



    Returns:

        A new Tensor with values of 1 where either `x` or `y` is 1, and 0 elsewhere.

    """

    return (x + y) > 0

