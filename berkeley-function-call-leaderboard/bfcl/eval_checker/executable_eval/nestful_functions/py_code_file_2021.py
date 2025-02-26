from typing import List



def convert_hex_to_utf8(hexdigits: str) -> str:

    """Converts a given string of hex characters to its corresponding UTF-8 encoded string.



    Args:

        hexdigits: The string to be converted.



    Returns:

        A string containing the UTF-8 encoded version of the input.

    """

    hex_digits: List[str] = hexdigits.split()

    ascii_chars: List[str] = []

    for hex_digit in hex_digits:

        ascii_char: str = chr(int(hex_digit, 16))

        ascii_chars.append(ascii_char)

    utf8_string: str = ''.join(ascii_chars)



    return utf8_string

