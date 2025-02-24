from typing import List



def min_lex_order(strings: List[str]) -> str:

    """Returns the string with the minimum lexicographic order from a list of strings.



    If the list is empty, returns an empty string. If two strings have the same lexicographic order,

    returns the string from the list that appears first in the list.



    Args:

        strings: A list of strings.



    Returns:

        The string with the minimum lexicographic order.

    """

    if not strings:

        return ''



    sorted_strings = sorted(strings)

    return sorted_strings[0]

