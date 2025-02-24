from typing import Union



def from_hex_string_to_int(hex_string: Union[str, bytes]) -> int:

    """Converts a hexadecimal string to an integer.



    Args:

        hex_string: A string of hexadecimal digits (0123456789abcdef or 0123456789ABCDEF),

        with or without a '0x' prefix.



    Returns:

        An integer.

    """

    return int(hex_string, base=16)

