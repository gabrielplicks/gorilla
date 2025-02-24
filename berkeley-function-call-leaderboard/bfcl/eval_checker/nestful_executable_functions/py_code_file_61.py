from typing import Tuple



def first_non_matching_index(string1: str, string2: str) -> int:

    """

    Returns the index of the first non-matching character between two strings.

    If the strings are identical, returns -1.



    Args:

        string1: The first string.

        string2: The second string.

    """

    for i in range(min(len(string1), len(string2))):

        if string1[i] != string2[i]:

            return i

    return -1

