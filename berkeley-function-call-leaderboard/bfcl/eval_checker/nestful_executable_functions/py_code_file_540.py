from typing import Union



def replace_wildcard(s: str, c: Union[str, int]) -> str:

    """Replaces the first occurrence of a single-character wildcard `*` with a specific character `c` in a string `s`.



    Args:

        s: The input string.

        c: The character to replace the wildcard with.

    """

    return s.replace('*', str(c), 1)

