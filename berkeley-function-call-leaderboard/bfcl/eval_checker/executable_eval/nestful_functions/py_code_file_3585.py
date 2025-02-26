import pickle

import numpy as np



def encode_array_py2(array: np.ndarray) -> str:

    """Encodes a Numpy array into a binary string using pickle for Python 2.



    Args:

        array: The Numpy array to be encoded.



    Returns:

        The encoded binary string.

    """

    list_to_encode = array.tolist()

    encoded_string = pickle.dumps(list_to_encode)

    encoded_string = str(encoded_string)



    return encoded_string



def encode_array_py3(array: np.ndarray) -> bytes:

    """Encodes a Numpy array into a binary string using pickle for Python 3.



    Args:

        array: The Numpy array to be encoded.



    Returns:

        The encoded binary string.

    """

    list_to_encode = array.tolist()

    encoded_string = pickle.dumps(list_to_encode)



    return encoded_string

