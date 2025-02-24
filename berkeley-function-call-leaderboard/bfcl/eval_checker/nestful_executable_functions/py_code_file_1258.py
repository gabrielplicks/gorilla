from typing import Optional



def left_trim(s: str, c: Optional[str] = None) -> str:

    """Removes all occurrences of the character `c` at the beginning of the string `s`.



    Args:

        s: The input string.

        c: The character to remove from the beginning of the string. If not provided,

           all leading whitespace characters will be removed.

    """

    while s and (c is None or s[0] == c):

        s = s[1:]

    return s

