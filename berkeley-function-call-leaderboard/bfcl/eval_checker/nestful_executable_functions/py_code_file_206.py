from typing import Union



def string_equals(string1: str, string2: str) -> bool:

    """Checks if two strings are equal.



    Args:

        string1: The first string to compare.

        string2: The second string to compare.



    Raises:

        ValueError: If both inputs are not strings.

    """

    if not isinstance(string1, str) or not isinstance(string2, str):

        raise ValueError("Both inputs must be strings.")

    return string1 == string2

