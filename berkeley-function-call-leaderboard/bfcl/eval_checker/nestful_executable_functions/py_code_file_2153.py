import base64

import numpy as np



def encode_numpy_array(array: np.ndarray) -> str:

    """Encodes a NumPy array to a base64-encoded string.



    Args:

        array: The input NumPy array.



    Returns:

        A base64-encoded string representing the array contents.

    """

    bytes_data = array.tobytes()

    base64_encoded = base64.b64encode(bytes_data)

    return base64_encoded

