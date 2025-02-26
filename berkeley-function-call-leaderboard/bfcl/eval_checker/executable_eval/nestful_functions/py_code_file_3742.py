from typing import Union



def encode_with_compact_string(s: str) -> str:

    """Encodes a string into its compact representation.

    Each character in `s` is represented by two hexadecimal digits.

    For example, the string `'abc'` is encoded as `'616263'`.

    Args:

        s: The input string to be encoded.

    """

    compact_string = ""

    for char in s:

        hex_value = hex(ord(char))

        compact_string += hex_value[2:]

    return compact_string

