from typing import Set



def are_overlapping(str1: str, str2: str) -> bool:

    """Determines whether the given `str1` and `str2` are overlapping.

    The two strings are considered overlapping if they share a common non-empty substring.

    Args:

        str1: The first string.

        str2: The second string.

    Returns:

        True if the strings are overlapping, False otherwise.

    """

    return set(str1) & set(str2) != set()

