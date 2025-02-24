from typing import Tuple



def compare(str1: str, str2: str) -> int:

    """Compares two strings, str1 and str2, based on the rules below.

    If str1 is lexicographically greater than str2, then return 1.

    If str1 is lexicographically less than str2, then return -1.

    Otherwise, return 0.

    Args:

        str1: The first string to compare.

        str2: The second string to compare.

    """

    for char1, char2 in zip(str1, str2):

        if char1 > char2:

            return 1

        elif char1 < char2:

            return -1

    if len(str1) < len(str2):

        return -1

    elif len(str1) > len(str2):

        return 1

    else:

        return 0

