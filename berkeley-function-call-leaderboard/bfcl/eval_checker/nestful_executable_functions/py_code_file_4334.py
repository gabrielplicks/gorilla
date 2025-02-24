from typing import Tuple



def little_endian_to_int(byte_string: bytes) -> int:

    """

    Converts a byte string into an integer using little-endian encoding.

    This function assumes that the byte string is in little-endian format,

    meaning that the most significant byte is at the end of the byte string.

    Args:

        byte_string: The byte string to convert.

    """

    result = 0



    for i in range(len(byte_string) - 1, -1, -1):

        result += byte_string[i] << (8 * (len(byte_string) - 1 - i))



    return result

