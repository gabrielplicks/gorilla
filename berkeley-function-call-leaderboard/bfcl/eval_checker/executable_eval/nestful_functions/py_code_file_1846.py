from typing import Union



def get_byte(num: int, n: int) -> str:

    """Returns the hexadecimal representation of the byte at the specified index `n` in the integer `num`.

    Args:

        num: The integer to extract the byte from.

        n: The index of the byte to extract.

    """

    masked_byte = (num >> (n * 8)) & 0xff

    return hex(masked_byte)[2:].zfill(2)

