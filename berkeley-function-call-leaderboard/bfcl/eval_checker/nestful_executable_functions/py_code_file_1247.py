from typing import Union



def color_string_to_int(color_string: str) -> str:

    """Converts a color string (RGB in hexadecimal) to a 24-bit integer represented in hexadecimal format with 6 digits.



    Args:

        color_string: The color string in hexadecimal format.



    Returns:

        The 24-bit integer represented in hexadecimal format with 6 digits.

    """

    color_int = int(color_string, 16)

    result = color_int & 0xFFFFFF

    return hex(result)[2:].zfill(6)

