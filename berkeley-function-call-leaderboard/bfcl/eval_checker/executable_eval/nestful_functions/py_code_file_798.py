from collections import Counter

from typing import AnyStr



def are_permutations(string1: AnyStr, string2: AnyStr) -> bool:

    """Determines whether one string is a permutation of the other, ignoring case.



    Args:

        string1: The first string.

        string2: The second string.



    Returns:

        A boolean value indicating whether the two strings are permutations of each other.

    """

    string1 = string1.lower()

    string2 = string2.lower()

    dict1 = Counter(string1)

    dict2 = Counter(string2)

    return dict1 == dict2

