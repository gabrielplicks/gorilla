from typing import List, Union



def add_hyphen(s: str) -> str:

    """Adds a hyphen between each pair of characters in a string.



    Args:

        s: The input string.



    Returns:

        The string with hyphens added between each pair of characters.

    """

    return '-'.join([c for c in s])

