from typing import Union



def decode_num(s: str) -> int:

    """Decodes a base 16 or base 10 number encoded as a string.

    Args:

        s: The input string.

    Raises:

        ValueError: If the input string is invalid.

    """

    if s.startswith("0x"):

        return int(s[2:], 16)

    else:

        return int(s, 10)

