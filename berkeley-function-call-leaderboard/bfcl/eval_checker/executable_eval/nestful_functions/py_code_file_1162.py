from typing import List



def altered_string(s: str) -> str:

    """

    Alters a string by keeping the first 3 characters as-is, and alternating the case of the remaining characters.

    If the length of `s` is less than 3, the function returns `s` as-is.



    Args:

        s: The input string.

    """

    return s[:3] + ''.join([c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(s[3:])])

