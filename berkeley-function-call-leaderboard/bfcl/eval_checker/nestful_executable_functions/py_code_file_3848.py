from typing import Tuple



def capitalize_second(string1: str, string2: str) -> str:

    """Concatenates two strings, with the first character of the second string capitalized.

    Args:

        string1: The first input string.

        string2: The second input string.

    """

    return string1 + string2[0].upper() + string2[1:]

