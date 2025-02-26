import numpy as np



def check_tensor_unique(tensor: np.ndarray) -> bool:

    """Checks whether a given tensor is unique (all values are distinct).



    Args:

        tensor: The tensor to check.



    Returns:

        True if the tensor is unique, False otherwise.

    """

    flattened_tensor = tensor.flatten()

    counts = np.unique(flattened_tensor, return_counts=True)[1]

    return np.all(counts == 1)

