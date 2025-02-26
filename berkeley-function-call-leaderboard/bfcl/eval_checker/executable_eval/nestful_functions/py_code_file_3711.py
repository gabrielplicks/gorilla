from typing import Union



def int_to_hex_string(num: Union[int, float]) -> str:

    """Converts an integer or float to its hexadecimal representation as a string.

    The string is zero-padded on the left to have a length of at least 8 digits.

    Args:

        num: The integer or float to be converted.

    """

    hex_str = hex(int(num))[2:].upper()  # Remove the leading '0x'

    return f'{hex_str:0>8}'  # Format the string with zero-padding

