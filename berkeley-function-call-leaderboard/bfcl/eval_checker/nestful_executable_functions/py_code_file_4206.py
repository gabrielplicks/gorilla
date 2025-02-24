import secrets

from string import hexdigits

from typing import List



def rand_hex_str(length: int) -> str:

    """Generates a random hexadecimal string of the given length.

    The string consists of uppercase A-F and 0-9 characters.

    Args:

        length: The length of the hexadecimal string.

    """

    rand_hex_chars: List[str] = [secrets.choice(hexdigits) for _ in range(length)]

    return ''.join(rand_hex_chars)

