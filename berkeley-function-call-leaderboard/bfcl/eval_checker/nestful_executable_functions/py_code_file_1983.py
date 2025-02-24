from typing import Union



def convert_to_big_endian(x: Union[int, bytes]) -> int:

    """Converts a 32-bit integer `x` (represented in little-endian byte order) to a 32-bit integer in big-endian byte order.

    Args:

        x: The 32-bit integer to be converted.

    """

    byte1 = x & 0x000000FF

    byte2 = (x & 0x0000FF00) >> 8

    byte3 = (x & 0x00FF0000) >> 16

    byte4 = (x & 0xFF000000) >> 24

    big_endian_value = (byte1 << 24) | (byte2 << 16) | (byte3 << 8) | byte4



    return big_endian_value

