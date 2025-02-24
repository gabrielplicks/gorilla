import numpy as np



def list_to_tensor(coordinates: List[List[int]]) -> List[np.ndarray]:

    """Converts a list of coordinates into a list of tensors.



    Args:

        coordinates: A list of coordinates.



    Returns:

        A list of tensors.

    """

    return [np.array(coordinate) for coordinate in coordinates]



def tensor_reduction(tensors: List[np.ndarray]) -> np.ndarray:

    """Converts a list of tensors into a single tensor.



    Args:

        tensors: A list of tensors.



    Returns:

        A single tensor.

    """

    return np.array(tensors)

