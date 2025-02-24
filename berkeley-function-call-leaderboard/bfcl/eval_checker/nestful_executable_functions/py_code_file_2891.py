from typing import Iterable



def all_characters_same(s: str) -> bool:

    """Checks if all characters in a string are the same.



    Args:

        s: The input string.



    Returns:

        True if all characters in the string are the same, False otherwise.

    """

    if len(s) == 0:

        return True

    return all(c == s[0] for c in s)

