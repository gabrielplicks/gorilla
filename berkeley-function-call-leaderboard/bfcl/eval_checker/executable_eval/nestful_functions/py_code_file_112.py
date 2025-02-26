import os

from typing import Tuple



def read_binary_and_name(path: str) -> Tuple[bytes, str]:

    """Reads the contents of the given path into a byte array and returns a tuple containing the byte array and the file name of the path (without its extension).



    Args:

        path: The path to the file.



    Returns:

        A tuple containing the byte array and the file name of the path (without its extension).

    """

    filename = os.path.basename(path)  # Get the file name without the extension

    file_extension = os.path.splitext(filename)[1]  # Get the extension

    filename_without_extension = os.path.splitext(filename)[0]  # Get the file name without its extension



    with open(path, 'rb') as file:

        byte_array = file.read()



    return byte_array, filename_without_extension

