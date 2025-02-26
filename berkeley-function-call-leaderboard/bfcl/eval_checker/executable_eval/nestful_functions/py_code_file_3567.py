from typing import List



def convert_to_ascii_codes(text: str) -> List[int]:

    """Converts a string to a list of ASCII executable_functions.

    Args:

        text: The input string.

    Returns:

        A list of integers representing the ASCII values of the characters in the input string.

    """

    return [ord(c) for c in text]

