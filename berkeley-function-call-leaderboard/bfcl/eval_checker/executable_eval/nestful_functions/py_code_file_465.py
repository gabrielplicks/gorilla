import numpy as np



def tensor_operation(tensor: np.ndarray) -> None:

    """Applies a mathematical expression to a tensor and modifies it in-place.



    Args:

        tensor: The input tensor of unknown shape and length.

    """

    tensor = np.exp(tensor) + 1

    tensor[:] = tensor

