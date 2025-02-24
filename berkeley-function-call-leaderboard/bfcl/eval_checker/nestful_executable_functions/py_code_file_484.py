import torch

from typing import Tuple



def y(x: torch.Tensor) -> torch.Tensor:

    """Computes the reciprocal of a tensor.



    Args:

        x: The input tensor.



    Returns:

        The reciprocal of the input tensor.

    """

    return 1 / x



x = torch.tensor([1, 2, 3])

result = y(x)

print(result)

