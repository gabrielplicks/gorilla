from typing import Union



def swap_bytes(x: Union[int, bytes]) -> int:

    """Swaps bytes in a 32-bit integer using bitwise operations.



    Args:

        x: The 32-bit integer to swap bytes in.



    Returns:

        The 32-bit integer with its bytes swapped.

    """

    byte1 = (x >> 24) & 0xFF

    byte2 = (x >> 16) & 0xFF

    byte3 = (x >> 8) & 0xFF

    byte4 = x & 0xFF

    return (byte4 << 24) | (byte3 << 16) | (byte2 << 8) | byte1

