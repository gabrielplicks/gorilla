from typing import Union



def concatenate_and_limit(s1: str, s2: str, n: Union[int, float]) -> str:

    """Concatenates the given strings `s1` and `s2` and returns the concatenated string.

    If the concatenated string is longer than `n`, only return the first `n` characters of the concatenated string.

    Args:

        s1: The first string to concatenate.

        s2: The second string to concatenate.

        n: The maximum length of the returned string.

    """

    return (s1 + s2)[:n]

