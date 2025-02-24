from typing import AnyStr



def pattern_counter(pattern: AnyStr, string: AnyStr) -> int:

    """Counts the total number of times a string pattern appears in a larger string.



    Args:

        pattern: The string pattern to search for.

        string: The larger string to search in.



    Returns:

        The total number of times the pattern appears in the larger string.

    """

    return string.count(pattern)

