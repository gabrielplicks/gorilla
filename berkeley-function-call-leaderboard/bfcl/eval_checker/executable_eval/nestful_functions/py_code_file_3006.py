from typing import Union



def hex_to_32bit_int(hex_string: str) -> int:

    """Converts a 32-bit hexadecimal string to a decimal integer.



    Args:

        hex_string: A hexadecimal string.



    Returns:

        The decimal integer equivalent of the hexadecimal string.

    """

    decimal_number = int(hex_string, 16)

    return decimal_number * (2 ** 32)

