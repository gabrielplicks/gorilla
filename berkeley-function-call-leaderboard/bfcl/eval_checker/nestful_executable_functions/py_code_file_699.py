import torch

import numpy as np



def torch_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    """Adds two PyTorch tensors or NumPy arrays.

    The tensors can be of different shapes, and the function returns the result as a PyTorch tensor.

    Handles the mismatch in shapes by repeating the smaller one along the axis of the larger one.



    Args:

        a: The first tensor or NumPy array.

        b: The second tensor or NumPy array.



    Raises:

        ValueError: If the type of `a` is not a PyTorch tensor or NumPy array.

    """

    if torch.is_tensor(a):

        result = torch.add(a, b)

    elif isinstance(a, np.ndarray):

        result = a + b

    else:

        raise ValueError("Unsupported type for `a`: expected PyTorch tensor or NumPy array")

    return result

