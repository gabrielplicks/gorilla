from typing import List



def encode_ascii(s: str) -> str:

    """Encodes a string `s` into a string containing the ASCII values of each character in `s`.



    Args:

        s: The input string.

    """

    return ''.join(str(ord(c)) for c in s)

