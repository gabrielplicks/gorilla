import numpy as np



def array_to_bytes(array: np.ndarray) -> bytes:

    """Converts a NumPy array to a bytes representation that can be directly written to a binary file.



    Args:

        array: The NumPy array to be converted.



    Returns:

        A bytes representation of the input NumPy array.

    """

    return array.tobytes()

