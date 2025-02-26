from typing import Union



def ascii_hex_encoding(string: Union[str, bytes]) -> str:

    """Converts a string to its ASCII hex encoding.



    Args:

        string: The input string.



    Returns:

        The ASCII hex encoding of the input string.

    """

    ascii_hex_codes = []



    for char in string:

        ascii_code = ord(char)

        ascii_hex_codes.append(hex(ascii_code)[2:])



    return '-'.join(ascii_hex_codes)

