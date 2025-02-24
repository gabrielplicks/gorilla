import numpy as np



def binary_file_to_byte_array(file_path: str) -> np.ndarray:

    """Reads a binary file and converts it to a byte array.



    Args:

        file_path: The path to the binary file.



    Returns:

        A NumPy array containing the byte data from the binary file.

    """

    with open(file_path, "rb") as f:

        data = f.read()

    return np.array(list(data))

