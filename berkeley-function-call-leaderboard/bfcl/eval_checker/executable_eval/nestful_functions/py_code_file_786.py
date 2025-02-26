from typing import Union



def to_hex(char: Union[str, bytes]) -> str:

    """Converts a string containing an ASCII character to a hexadecimal representation of its ASCII value.

    Args:

        char: The ASCII character to convert.

    """

    if char == '\0':

        return '0x0'

    return hex(ord(char))

