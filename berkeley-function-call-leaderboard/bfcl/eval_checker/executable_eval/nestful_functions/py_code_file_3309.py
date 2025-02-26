from typing import Tuple



def replace_first_star(s1: str, s2: str) -> str:

    """Replaces the first occurrence of a `*` in `s1` with `s2`.

    Args:

        s1: The input string.

        s2: The string to replace the first `*` with.

    """

    for i, c in enumerate(s1):

        if c == "*":

            return s1[:i] + s2 + s1[i + 1:]

    return s1

