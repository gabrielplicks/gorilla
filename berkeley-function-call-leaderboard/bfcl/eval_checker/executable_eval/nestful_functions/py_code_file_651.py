from typing import List



def string_to_ascii(string: str) -> List[int]:

    """Converts a string to its corresponding ASCII executable_functions and returns them as a list of integers.



    Args:

        string: The input string to convert to ASCII executable_functions.



    Returns:

        A list of integers representing the ASCII executable_functions of the input string.

    """

    ascii_codes = []

    for char in string:

        ascii_codes.append(ord(char))

    return ascii_codes

