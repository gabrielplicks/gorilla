import zlib



def compress_data(data: str) -> bytes:

    """Compresses a given string of data using the zlib module's compress function.



    Args:

        data: The string of data to compress.



    Raises:

        TypeError: If the input is not a string.

    """

    if not isinstance(data, str):

        raise TypeError("Input must be a string")

    return zlib.compress(data.encode("utf-8"))

