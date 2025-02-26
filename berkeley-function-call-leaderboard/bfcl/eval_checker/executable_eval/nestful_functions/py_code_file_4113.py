from typing import Tuple



def compare_strings_in_constant_time(string1: str, string2: str) -> bool:

    """Compares two strings in constant time to prevent timing attacks.



    Args:

        string1: The first input string.

        string2: The second input string.



    Returns:

        A boolean indicating whether the strings are equal.

    """

    total = 0

    for char1, char2 in zip(string1, string2):

        total += ord(char1) ^ ord(char2)

    return total == 0

