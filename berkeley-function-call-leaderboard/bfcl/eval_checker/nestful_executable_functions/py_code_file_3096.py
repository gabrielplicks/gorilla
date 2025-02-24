from typing import List



def replace_chars_with_ascii(input_string: str) -> str:

    """Replaces each character in a given string with its ASCII character code.



    Args:

        input_string: The input string.



    Returns:

        A comma-separated list of ASCII character executable_functions.

    """

    ascii_codes: List[str] = []

    for char in input_string:

        ascii_codes.append(str(ord(char)))



    return ', '.join(ascii_codes)

