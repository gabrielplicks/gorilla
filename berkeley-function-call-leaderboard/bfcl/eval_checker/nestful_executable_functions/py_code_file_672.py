from typing import Union



def string_to_utf8_hex(s: Union[str, bytes]) -> str:

    """Converts a string `s` to a UTF-8 encoded byte array and then converts the bytes into their hexadecimal representation, returning a string.



    Args:

        s: The input string.



    Returns:

        The hexadecimal representation of the UTF-8 encoded byte array.

    """

    return s.encode('utf-8').hex()

