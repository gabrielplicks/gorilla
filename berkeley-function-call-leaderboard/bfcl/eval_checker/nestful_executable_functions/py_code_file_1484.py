from typing import AnyStr



def substring_count(s: AnyStr, t: AnyStr) -> int:

    """Calculates the number of occurrences of `t` as a substring in `s`.

    The function ignores case sensitivity and any leading and trailing spaces.

    Args:

        s: The input string.

        t: The substring to search for.

    """

    s = s.strip()

    t = t.strip()

    s = s.lower()

    t = t.lower()

    return s.count(t)

