from typing import List



def parse_terminal_code(terminal_code: List[str]) -> str:

    """Parses a given terminal code and returns the associated string.

    The terminal code is represented by a series of 0s and 1s.

    For each 0, append the letter 'B' to the string. For each 1, append the letter 'A' to the string.

    The length of the terminal code is always a multiple of 4.

    Args:

        terminal_code: The terminal code to parse.

    """

    result = ""

    for i, bit in enumerate(terminal_code):

        letter = 'B' if bit == '0' else 'A'

        result += letter

        result += f"({i}: {bit})"

    return result

