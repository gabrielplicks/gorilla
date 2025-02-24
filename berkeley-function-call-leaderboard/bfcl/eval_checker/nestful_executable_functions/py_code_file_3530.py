import re

from typing import Union



def is_valid_hex_string(string: str) -> bool:

    """

    Checks if a given string is a valid hexadecimal string. The function returns True if the string is

    a valid hexadecimal string, otherwise it returns False.



    Args:

        string: The string to check if it is a valid hexadecimal string.



    Returns:

        True if the string is a valid hexadecimal string, otherwise False.

    """

    try:

        assert isinstance(string, str)

        assert len(string) > 0

    except AssertionError:

        return False



    return bool(re.match(r'^0[xX][0-9a-fA-F]+$', string))

