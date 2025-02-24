from typing import Dict, List



def group_by_first_char(strings: List[str]) -> Dict[str, List[str]]:

    """Groups a list of strings by their first character, similar to a phone book.

    Args:

        strings: A list of strings to be grouped.

    Returns:

        A dictionary with keys as letters, and values as lists of strings starting with the same letter, sorted in lexicographical order.

    """

    grouped = {}

    for s in strings:

        first_char = s[0]

        if first_char not in grouped:

            grouped[first_char] = []

        grouped[first_char].append(s)

    return grouped

