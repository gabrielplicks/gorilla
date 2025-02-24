from typing import Union



def convert_to_hex(n: int) -> str:

    """Converts an integer into a string representing the integer in hexadecimal format.



    Args:

        n: The integer to be converted.



    Returns:

        A string representing the integer in hexadecimal format.

    """

    return hex(n)[2:] if n >= 0 else '-' + hex(abs(n))[2:]

