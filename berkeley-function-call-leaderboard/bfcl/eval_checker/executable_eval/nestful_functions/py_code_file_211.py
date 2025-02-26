from typing import Union



def to_hex_binary(string: Union[str, bytes]) -> str:

    """Converts the given string to its binary representation in hexadecimal format.



    Args:

        string: The input string.



    Returns:

        The binary representation of the input string in hexadecimal format.

    """

    ascii_string = string.encode('ascii')

    hex_string = ascii_string.hex()

    return hex_string

