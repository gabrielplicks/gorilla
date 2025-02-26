from typing import BinaryIO



def read_first_10_bytes(file_path: str) -> bytes:

    """Reads the first 10 bytes of a file.



    Args:

        file_path: The path to the file.



    Returns:

        The first 10 bytes of the file as a byte string.

    """

    with open(file_path, "rb") as file:

        return file.read(10)

