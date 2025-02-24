from typing import Union



def int_to_hex_with_prefix(number: Union[int, float]) -> str:

    """Converts a positive integer number to its hexadecimal representation with a "0x" prefix.



    Args:

        number: The positive integer number to be converted.



    Returns:

        The hexadecimal representation of the number with a "0x" prefix.

    """

    hexadecimal = hex(number)

    return "0x" + hexadecimal[2:]

