from typing import Optional



def remove_character(s: str, c: str) -> str:

    """Removes all occurrences of the character `c` from the string `s`.

    If the string `s` is empty, the function returns a string containing only a single space.

    Args:

        s: The input string.

        c: The character to be removed.

    """

    if s == '':

        return ' '

    return s.replace(c, '')

