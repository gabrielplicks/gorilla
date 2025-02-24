from typing import List



def print_printables(s: str) -> str:

    """

    Returns a string of all printable characters from the input string `s`.

    Args:

        s: The input string.

    """

    printable_chars: List[str] = []

    for char in s:

        ascii_code = ord(char)

        if 32 <= ascii_code <= 126:

            printable_chars.append(char)

    return ''.join(printable_chars)

