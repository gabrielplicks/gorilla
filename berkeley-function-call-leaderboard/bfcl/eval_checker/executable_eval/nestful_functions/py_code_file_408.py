from typing import Union



def substring_from_index(s: str, i: Union[int, float]) -> str:

    """

    Returns the substring of `s` starting at index `i` and ending at the end of the string.

    If `i` is out of bounds, an empty string is returned.



    Args:

        s: The input string.

        i: The starting index.

    """

    if i < 0 or i > len(s) - 1:

        return ""

    return s[i:]

