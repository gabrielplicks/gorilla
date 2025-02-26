from itertools import groupby

from typing import Iterable



def remove_consecutive_duplicates(text: str) -> str:

    """Removes all consecutive duplicate characters in a string and replaces them with a single instance of the character.



    Args:

        text: The input string.



    Returns:

        The string with consecutive duplicate characters removed.

    """

    groups = groupby(text)

    return ''.join(group[0] for group in groups)

