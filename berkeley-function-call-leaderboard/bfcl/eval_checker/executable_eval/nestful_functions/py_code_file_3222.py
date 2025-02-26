from typing import Union



def byte_order(num: Union[int, float]) -> int:

    """Reverses the byte order of a 32-bit unsigned integer.



    Args:

        num: A 32-bit unsigned integer.



    Returns:

        A 32-bit unsigned integer with reversed byte order.

    """

    byte1 = (num & 0xFF000000) >> 24

    byte2 = (num & 0x00FF0000) >> 16

    byte3 = (num & 0x0000FF00) >> 8

    byte4 = (num & 0x000000FF)

    return byte4 << 24 | byte3 << 16 | byte2 << 8 | byte1

