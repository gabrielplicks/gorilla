from typing import Any



def to_utf8(string: str) -> bytes:

    """Converts a string to UTF-8 encoding.



    Args:

        string: The input string to be converted to UTF-8.



    Raises:

        TypeError: If the input is not a string.

    """

    if not isinstance(string, str):

        raise TypeError(f'"{string}" is not a string')



    return bytes(string, 'utf-8')

