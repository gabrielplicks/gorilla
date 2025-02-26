import numpy as np



def normalize_numpy(values: list[float]) -> np.ndarray:

    """Normalizes a given list of float values and returns the result in a numpy array.

    Args:

        values: The list of float values to normalize.

    """

    array = np.array(values)

    return (array - array.min()) / (array.max() - array.min())

