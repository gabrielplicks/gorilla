import numpy as np

from typing import Any



def find_index_of_max(input_tensor: np.ndarray) -> np.ndarray:

    """

    Returns the index of the maximum value in the last dimension of the input tensor.

    Args:

        input_tensor: The input tensor.

    Raises:

        ValueError: If the input tensor has less than 2 dimensions.

    """

    if len(input_tensor.shape) < 2:

        raise ValueError("The input tensor must have at least 2 dimensions.")



    index = np.argmax(input_tensor, axis=-1)

    return index

